from pathlib import Path
from typing import Union

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from model.tacotron import CBHG
from utils.config import Config


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


class DurationPredictor(nn.Module):

    def __init__(self, in_dims, conv_dims=256, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(in_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self, x, alpha=1.0):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
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
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = DurationPredictor(embed_dims,
                                          conv_dims=durpred_conv_dims,
                                          rnn_dims=durpred_rnn_dims,
                                          dropout=durpred_dropout)
        self.prenet = CBHG(K=prenet_k,
                           in_channels=embed_dims,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, embed_dims],
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
        dur_hat = self.dur_pred(x)
        dur_hat = dur_hat.squeeze()

        bs = dur.shape[0]
        ends = torch.cumsum(dur.float(), dim=1)
        mids = ends - dur.float() / 2.

        x = x.transpose(1, 2)
        x_p = self.prenet(x)
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

        # x = self.lr(x, dur)
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
        dur = self.dur_pred(x, alpha=alpha)
        dur = dur.squeeze(2)

        bs = dur.shape[0]
        ends = torch.cumsum(dur, dim=1)
        mids = ends - dur / 2.

        x = x.transpose(1, 2)
        x_p = self.prenet(x)
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