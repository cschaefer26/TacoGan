from pathlib import Path
from typing import Union

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.config import Config


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, activation=None)

        # Fix the highway input if necessary
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x):
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x) # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways: x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]

class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dur):
        return self.expand(x, dur)
     
    @staticmethod
    def build_index(duration, x):
        duration[duration < 0] = 0
        tot_duration = duration.cumsum(1).detach().cpu().numpy().astype('int')
        max_duration = int(tot_duration.max().item())
        index = np.zeros([x.shape[0], max_duration, x.shape[2]], dtype='long')

        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                pos1 = tot_duration[i, j]
                index[i, pos:pos1, :] = j
                pos = pos1
            index[i, pos:, :] = j
        return torch.LongTensor(index).to(duration.device)

    def expand(self, x, dur):
        idx = self.build_index(dur, x)
        y = torch.gather(x, 1, idx)
        return y


class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class DurationPredictor(nn.Module):

    def __init__(self, in_dims, conv_dims=256, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(in_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
            #BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
        ])

        self.lin = nn.Linear(conv_dims, 1)
        self.dropout = dropout

    def forward(self, x, alpha=1.0):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = self.lin(x)
        return x / alpha


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, activation=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        x = self.bnorm(x)
        return x


class ForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_chars,
                 durpred_conv_dims,
                 durpred_rnn_dims,
                 durpred_dropout,
                 rnn_dim,
                 prenet_k,
                 prenet_dims,
                 postnet_k,
                 postnet_dims,
                 highways,
                 dropout,
                 n_mels):

        super().__init__()
        self.rnn_dim = rnn_dim
        self.pre = PreNet(embed_dims)
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = DurationPredictor(2*prenet_dims,
                                          conv_dims=durpred_conv_dims,
                                          rnn_dims=durpred_rnn_dims,
                                          dropout=durpred_dropout)
        self.prenet = CBHG(K=prenet_k,
                           in_channels=128,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, 128],
                           num_highways=highways)
        self.lstm = nn.LSTM(2 * prenet_dims,
                            rnn_dim,
                            batch_first=True,
                            bidirectional=True)
        self.lin = torch.nn.Linear(2 * rnn_dim, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.postnet = CBHG(K=postnet_k,
                            in_channels=n_mels,
                            channels=postnet_dims,
                            proj_channels=[postnet_dims, n_mels],
                            num_highways=highways)
        self.dropout = dropout
        self.post_proj = nn.Linear(2 * postnet_dims, n_mels, bias=False)

    def forward(self, x, mel, dur):
        if self.training:
            self.step += 1

        x = self.embedding(x)
        x = self.pre(x)

        bs = dur.shape[0]
        ends = torch.cumsum(dur.float(), dim=1)
        mids = ends - dur.float() / 2.

        x = x.transpose(1, 2)
        x_p = self.prenet(x)

        dur_hat = self.dur_pred(x_p)
        dur_hat = dur_hat.squeeze()

        device = next(self.parameters()).device
        mel_len = mel.shape[-1]
        seq_len = mids.shape[1]

        t_range = torch.arange(0, mel_len).long().to(device)
        t_range = t_range.unsqueeze(0)
        t_range = t_range.expand(bs, mel_len)
        t_range = t_range.unsqueeze(-1)
        t_range = t_range.expand(bs, mel_len, seq_len)

        mids = mids.unsqueeze(1)
        diff = t_range - mids
        logits = -diff ** 2 / 10. - 1e-9
        weights = torch.softmax(logits, dim=2)
        x = torch.einsum('bij,bjk->bik', weights, x_p)


        #x = self.lr(x, dur)
        x, _ = self.lstm(x)
        x = F.dropout(x,
                      p=self.dropout,
                      training=self.training)
        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = self.pad(x_post, mel.size(2))
        x = self.pad(x, mel.size(2))
        return x, x_post, dur_hat

    def generate(self, x, alpha=1.0):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        x = self.embedding(x)
        x = self.pre(x)

        x = x.transpose(1, 2)

        x_p = self.prenet(x)

        dur = self.dur_pred(x_p, alpha=alpha)
        dur = dur.squeeze(2)
        bs = dur.shape[0]
        ends = torch.cumsum(dur, dim=1)
        mids = ends - dur / 2.

        device = next(self.parameters()).device
        mel_len = int(torch.sum(dur))
        seq_len = mids.shape[1]

        t_range = torch.arange(0, mel_len).long().to(device)
        t_range = t_range.unsqueeze(0)
        t_range = t_range.expand(bs, mel_len)
        t_range = t_range.unsqueeze(-1)
        t_range = t_range.expand(bs, mel_len, seq_len)

        mids = mids.unsqueeze(1)
        diff = t_range - mids
        logits = -diff ** 2 / 10. - 1e-9
        weights = torch.softmax(logits, dim=2)
        x = torch.einsum('bij,bjk->bik', weights, x_p)

        x, _ = self.lstm(x)
        x = F.dropout(x,
                      p=self.dropout,
                      training=self.training)
        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x, x_post, dur = x.squeeze(), x_post.squeeze(), dur.squeeze()
        x = x.cpu().data.numpy()
        x_post = x_post.cpu().data.numpy()
        dur = dur.cpu().data.numpy()

        return x, x_post, dur

    def pad(self, x, max_len):
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', 0.0)
        return x

    def get_step(self):
        return self.step.data.item()

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    @classmethod
    def from_config(cls, cfg: Config):
        return ForwardTacotron(
                 embed_dims=cfg.embed_dims,
                 num_chars=len(cfg.symbols)+1,
                 durpred_conv_dims=cfg.durpred_conv_dims,
                 durpred_rnn_dims=cfg.durpred_rnn_dims,
                 durpred_dropout=cfg.durpred_dropout,
                 rnn_dim=cfg.rnn_dim,
                 prenet_k=cfg.prenet_k,
                 prenet_dims=cfg.prenet_dims,
                 postnet_k=cfg.postnet_k,
                 postnet_dims=cfg.postnet_dims,
                 highways=cfg.highways,
                 dropout=cfg.dropout,
                 n_mels=cfg.n_mels)