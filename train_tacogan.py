import torch
import torch.nn.functional as F
import traceback
from torch import optim
from torch.utils.data.dataloader import DataLoader

from audio import Audio
from dataset import new_audio_datasets
from model.tacotron import Tacotron
from utils.display import plot_mel, plot_attention
from utils.io import read_config
from utils.paths import Paths


class Session:

    def __init__(self,
                 r: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: DataLoader) -> None:
        self.r = r
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = Paths()
        self.audio = Audio(cfg)

    def train(self, model, optimizer):
        for session_params in self.cfg['training_schedule']:
            r, lr, max_step, bs = session_params
            if model.step < max_step:
                train_set, val_set = new_audio_datasets(
                    paths=paths, batch_size=bs, r=r, cfg=cfg)
                session = Session(
                    r=r, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model, optimizer, session):
        model.train()
        model.r = session.r
        device = next(model.parameters()).device

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        for epoch in range(1000):

            for i, (seqs, mels, stops, ids, lens) in enumerate(session.train_set, 1):

                seqs, mels, stops = seqs.to(device), mels.to(device), stops.to(device)
                mels = mels.transpose(1, 2)
                lin_mels, post_mels, att = model(seqs, mels)
                lin_loss = F.l1_loss(lin_mels, mels)
                post_loss = F.l1_loss(post_mels, mels)
                loss = lin_loss + post_loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'{int(model.step)} {float(loss)}')

                if model.step % 2 == 0:
                    print(f'evaluating at step {model.step}')
                    self.evaluate(model, session.val_set)

            if model.step > session.max_step:
                return

    def evaluate(self, model, val_set):
        model.eval()
        val_loss = 0
        for i, (seqs, mels, stops, ids, lens) in enumerate(val_set, 1):
            with torch.no_grad():
                lin_mels, post_mels, att = model(seqs)
                lin_loss = F.l1_loss(lin_mels, mels)
                post_loss = F.l1_loss(post_mels, mels)
                val_loss += lin_loss + post_loss
            if i == 0:
                try:
                    seq = seqs[0].tolist()
                    _, m_gen, _ = model.generate(seq)
                    plot_mel(m_gen, f'/tmp/mel_{int(model.step)}_pred')
                except Exception:
                    traceback.print_exc()
                sample_mel = mels[0].detach()[:, :600].numpy()
                sample_pred = post_mels[0].detach()[:, :600].numpy()
                sample_att = att[0].detach().numpy()
                plot_mel(sample_pred, f'/tmp/mel_{int(model.step)}_gta')
                plot_mel(sample_mel, f'/tmp/mel_{int(model.step)}_target')
                plot_attention(sample_att, f'/tmp/att_{int(model.step)}')
        val_loss /= len(val_set)
        return val_loss



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cfg = read_config('config.yaml')
    paths = Paths()
    fft_bins = cfg['n_fft'] // 2 + 1

    print('\nInitialising Tacotron Model...\n')
    tacotron = Tacotron(embed_dims=256,
                        num_chars=len(cfg['symbols'])+1,
                        encoder_dims=128,
                        decoder_dims=256,
                        n_mels=80,
                        fft_bins=80,
                        postnet_dims=128,
                        encoder_K=16,
                        lstm_dims=512,
                        postnet_K=8,
                        num_highways=4,
                        dropout=0.5,
                        stop_threshold=-3.4).to(device)

    optimizer = optim.Adam(tacotron.parameters())

    trainer = Trainer(cfg)
    trainer.train(tacotron, optimizer)