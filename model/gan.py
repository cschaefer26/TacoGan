import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

from torch.nn.modules.dropout import Dropout
from torch.nn.modules.rnn import GRU, LSTM

from utils.config import Config


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.dropout = Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bnorm(x)
        x = self.dropout(x)
        return x


class Generator(nn.Module):

    def __init__(self, n_mels, conv_dim, rnn_dim, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, conv_dim, 5, activation=torch.tanh, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.tanh, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.tanh, dropout=dropout)
        ])
        self.lstm = LSTM(n_mels, rnn_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * rnn_dim, n_mels)

    def forward(self, x):
        #x = x.transpose(1, 2)
        #for conv in self.convs:
        #    x = conv(x)
        #x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, n_mels, conv_dim, rnn_dim, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, conv_dim, 5, activation=torch.tanh, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.tanh, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.tanh, dropout=dropout)
        ])
        self.lstm = LSTM(n_mels, rnn_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * rnn_dim, 1)

    def forward(self, x):
        #x = x.transpose(1, 2)
        #for conv in self.convs:
        #    x = conv(x)
        #x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x



class GAN(nn.Module):

    def __init__(self, n_mels, conv_dim, rnn_dim, dropout=0.5):
        super().__init__()
        self.generator = Generator(
            n_mels, conv_dim, rnn_dim, dropout=dropout)
        self.discriminator = Discriminator(
            n_mels, conv_dim, rnn_dim, dropout=dropout)

    def forward(self, x):
        x_gen = self.generator(x)
        x_disc = self.discriminator(x_gen)
        return x_gen, x_disc

    def load(self, path: Union[str, Path]):
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        torch.save(self.state_dict(), path)

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def from_config(cls, cfg: Config) -> 'GAN':
        return GAN(n_mels=cfg.n_mels,
                   conv_dim=cfg.gan_conv_dim,
                   rnn_dim=cfg.gan_rnn_dim,
                   dropout=cfg.gan_dropout)


if __name__ == '__main__':

    x = np.zeros((2, 100, 80))
    x = torch.tensor(x).float()
    gan = GAN(80, 256, 256)
    gan(x)