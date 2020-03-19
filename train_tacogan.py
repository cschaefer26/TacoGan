import torch
import torch.nn.functional as F
import traceback
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio import Audio
from dataset import new_audio_datasets
from model.tacotron import Tacotron
from utils.decorators import ignore_exception
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
        self.writer = SummaryWriter(log_dir=cfg['log_dir'], comment='v1')


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
        model.r = session.r
        device = next(model.parameters()).device

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        for epoch in range(1000):

            for i, (seqs, mels, stops, ids, lens) in enumerate(session.train_set, 1):

                seqs, mels, stops = seqs.to(device), mels.to(device), stops.to(device)
                mels = mels.transpose(1, 2)
                model.train()
                lin_mels, post_mels, att = model(seqs, mels)
                lin_loss = F.l1_loss(lin_mels, mels)
                post_loss = F.l1_loss(post_mels, mels)
                loss = lin_loss + post_loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'{int(model.step)} {float(loss)}')

                self.writer.add_scalar('Loss/train', loss, global_step=model.step)
                if model.step % 1 == 0:
                    val_loss = self.evaluate(model, session.val_set)
                    self.writer.add_scalar('Loss/val', val_loss, global_step=model.step)
                    print(f'step: {model.step} val_loss: {val_loss}')

            if model.step > session.max_step:
                return

    def evaluate(self, model, val_set) -> float:
        model.eval()
        val_loss = 0
        for i, batch in enumerate(val_set, 1):
            seqs, mels, stops, ids, lens = batch
            with torch.no_grad():
                mels = mels.transpose(1, 2)
                pred = model(seqs, mels)
                lin_mels, post_mels, att = pred
                lin_loss = F.l1_loss(lin_mels, mels)
                post_loss = F.l1_loss(post_mels, mels)
                val_loss += lin_loss + post_loss
            if i == 1:
                self.generate_samples(model, batch, pred)

        val_loss /= len(val_set)
        return float(val_loss)

    @ignore_exception
    def generate_samples(self, model, batch, pred):
        seqs, mels, stops, ids, lens = batch
        lin_mels, post_mels, att = pred
        mel_sample = mels[0].detach()[:, :lens[0]].numpy()
        gta_sample = post_mels[0].detach()[:, :lens[0]].numpy()
        target_mel = plot_mel(mel_sample)
        gta_mel = plot_mel(gta_sample)
        self.writer.add_figure('Mel/target', target_mel)
        self.writer.add_figure('Mel/ground_truth_aligned', gta_mel)
        seq = seqs[0].tolist()
        _, m_gen, _ = model.generate(seq)
        gen_mel = plot_mel(m_gen)
        self.writer.add_figure('Mel/generated', gen_mel)


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