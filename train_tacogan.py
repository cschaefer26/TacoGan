import torch
import torch.nn.functional as F
import traceback
import os
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio import Audio
from dataset import new_audio_datasets
from model.io import save_model, load_model
from model.tacotron import Tacotron
from utils.config import Config
from utils.decorators import ignore_exception
from utils.display import plot_mel, plot_attention
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

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths = Paths()
        self.audio = Audio(cfg)
        self.writer = SummaryWriter(log_dir=cfg.log_dir, comment='v1')
        self.steps_to_eval = cfg.steps_to_eval
        self.schedule = cfg.training_schedule

    def train(self, model, optimizer):
        for session_params in self.schedule:
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

                if model.step % cfg['steps_to_checkpoint'] == 0:
                    self.save_model(model, optimizer, step=model.get_step())

                self.writer.add_scalar('Loss/train', loss, model.get_step())
                if model.step % self.steps_to_eval == 0:
                    val_loss = self.evaluate(model, session.val_set)
                    self.writer.add_scalar('Loss/val', val_loss, model.step)

            # checkpoint latest model after epoch is finished
            self.save_model(model, optimizer)

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

    def save_model(self, model, optimizer, step=None):
        save_model(
            model=model, optimizer=optimizer,
            cfg=self.cfg, path=self.paths.ckpt/'latest_model.zip')
        if step is not None:
            save_model(
                model=model, optimizer=optimizer,
                cfg=self.cfg, path=self.paths.ckpt/f'model_step_{step}.zip')

    @ignore_exception
    def generate_samples(self, model, batch, pred):
        seqs, mels, stops, ids, lens = batch
        lin_mels, post_mels, att = pred
        mel_sample = mels[:, :lens[0]].transpose(1, 2)[0].detach().numpy()
        gta_sample = post_mels[0].detach()[:, :lens[0]].numpy()
        att_sample = att[0].detach().numpy()

        target_fig = plot_mel(mel_sample)
        gta_fig = plot_mel(gta_sample)
        att_fig = plot_attention(att_sample)
        self.writer.add_figure('Mel/target', target_fig, model.step)
        self.writer.add_figure('Mel/ground_truth_aligned', gta_fig, model.step)
        self.writer.add_figure('Attention/ground_truth_aligned', att_fig, model.step)

        target_wav = self.audio.griffinlim((mel_sample.T + 1.) / 2., 32)
        gta_wav = self.audio.griffinlim((gta_sample.T + 1.) / 2., 32)
        self.writer.add_audio(
            tag='Wav/target', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.audio.sample_rate)
        self.writer.add_audio(
            tag='Wav/ground_truth_aligned', snd_tensor=gta_wav,
            global_step=model.step, sample_rate=self.audio.sample_rate)

        seq = seqs[0].tolist()
        _, gen_sample, _ = model.generate(seq, steps=lens[0])
        gen_mel = plot_mel(gen_sample)
        self.writer.add_figure('Mel/generated', gen_mel, model.step)
        gen_wav = self.audio.griffinlim((gen_sample.T + 1.) / 2., 32)
        self.writer.add_audio(
            tag='Wav/generated', snd_tensor=gen_wav,
            global_step=model.step, sample_rate=self.audio.sample_rate)


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    paths = Paths()
    latest_ckpt = paths.ckpt/'latest_model.zip'
    if os.path.exists(latest_ckpt):
        print(f'Loading model from {latest_ckpt}')
        model, optimizer, cfg = load_model(latest_ckpt, device)
    else:
        print('\nInitialising new model from config...\n')
        cfg = Config.load('config.yaml')
        model = Tacotron.from_config(cfg).to(device)
        optimizer = optim.Adam(model.parameters())

    trainer = Trainer(cfg)
    trainer.train(model, optimizer)