import torch


class Discriminator(torch.nn.Module):

    def __init__(self, n_mels: int, lstm_dim: int) -> None:
        super().__init__()
        self.rnn = torch.nn.LSTM(
            n_mels, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x
