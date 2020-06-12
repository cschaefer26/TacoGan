import torch


class Aligner(torch.nn.Module):

    def __init__(self, n_mels: int, lstm_dim: int, num_symbols: int) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.rnn = torch.nn.LSTM(
            n_mels, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols)

    def forward(self, mels):
        if self.train:
            self.step += 1
        x, _ = self.rnn(mels)
        x = self.lin(x)
        return x

    def get_step(self):
        return self.step.data.item()