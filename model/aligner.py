import torch
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

from torch.nn.modules.dropout import Dropout
from torch.nn.modules.rnn import GRU

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


class Aligner(torch.nn.Module):

    def __init__(self, n_mels: int, lstm_dim: int, num_symbols: int) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, 256, 5, activation=torch.tanh, dropout=0.1),
            BatchNormConv(256, 256, 5, activation=torch.tanh, dropout=0.1),
            BatchNormConv(256, 256, 5, activation=torch.tanh, dropout=0.1)
        ])
        self.rnn = torch.nn.LSTM(
            256, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols)

    def forward(self, x):
        if self.train:
            self.step += 1
        x.transpose_(1, 2)
        for conv in self.convs:
            x = conv(x)
        x.transpose_(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x

    def get_step(self):
        return self.step.data.item()