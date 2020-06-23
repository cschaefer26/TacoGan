import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

from torch.nn.modules.dropout import Dropout

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


class Aligner(torch.nn.Module):

    def __init__(self, n_mels: int, conv_dim, lstm_dim: int, num_symbols: int, dropout=0) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, conv_dim, 5, activation=torch.relu, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.relu, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.relu, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.relu, dropout=dropout),
            BatchNormConv(conv_dim, conv_dim, 5, activation=torch.relu, dropout=dropout),
        ])
        self.rnn = torch.nn.LSTM(
            conv_dim, lstm_dim, batch_first=True, bidirectional=True)
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

    @classmethod
    def from_config(cls, cfg: Config):
        return Aligner(n_mels=cfg.n_mels,
                       lstm_dim=cfg.aligner_lstm_dim,
                       conv_dim=cfg.aligner_conv_dim,
                       num_symbols=len(cfg.symbols)+1,
                       dropout=cfg.aligner_dropout)