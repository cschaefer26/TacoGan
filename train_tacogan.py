import argparse
import time

import torch
import torch.nn.functional as F
import os
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio import Audio
from dataset import new_audio_datasets
from losses import MaskedL1
from model.io import save_model, load_model
from model.tacotron_new import Tacotron
from utils.common import Averager
from utils.config import Config
from utils.decorators import ignore_exception
from utils.display import plot_mel, plot_attention, display_params, stream
from utils.paths import Paths


class Session:

    def __init__(self,
                 index: int,
                 r: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: DataLoader) -> None:
        self.index = index
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
        self.criterion = MaskedL1()

    def train(self, model, optimizer):
        for i, session_params in enumerate(self.cfg.training_schedule, 1):
            r, lr, max_step, bs = session_params
            if model.step < max_step:
                train_set, val_set = new_audio_datasets(
                    paths=paths, batch_size=bs, r=r, cfg=cfg)
                session = Session(
                    index=i, r=r, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model, optimizer, session):
        model.r = session.r
        cfg = self.cfg
        device = next(model.parameters()).device
        display_params([
            ('Session', session.index), ('Reduction', session.r),
            ('Max Step', session.max_step), ('Learning Rate', session.lr),
            ('Batch Size', session.bs), ('Steps per Epoch', len(session.train_set))
        ])

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()

        while model.get_step() <= session.max_step:

            for i, (seqs, mels, stops, ids, lens) in enumerate(session.train_set):
                seqs, mels, stops, lens = \
                    seqs.to(device), mels.to(device), stops.to(device), lens.to(device)
                t_start = time.time()
                block_step = model.get_step() % cfg.steps_to_eval + 1

                model.train()
                lin_mels, post_mels, att = model(seqs, mels)

                lin_loss = self.criterion(lin_mels, mels, lens)
                post_loss = self.criterion(post_mels, mels, lens)

                loss = lin_loss + post_loss
                loss_avg.add(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                duration_avg.add(time.time() - t_start)
                steps_per_s = 1. / duration_avg.get()
                self.writer.add_scalar('Loss/train', loss, model.get_step())

                msg = f'{block_step}/{cfg.steps_to_eval} | Step: {model.get_step()} ' \
                      f'| {steps_per_s:#.2} steps/s | Loss: {loss_avg.get():#.4} '
                stream(msg)

                if model.step % cfg.steps_to_checkpoint == 0:
                    self.save_model(model, optimizer, step=model.get_step())

                if model.step % self.cfg.steps_to_eval == 0:
                    val_loss = self.evaluate(model, session.val_set, msg)
                    self.writer.add_scalar('Loss/val', val_loss, model.step)
                    self.save_model(model, optimizer, step=model.get_step())
                    stream(msg + f'| Val Loss: {float(val_loss):#0.4} \n')
                    loss_avg.reset()
                    duration_avg.reset()

            if model.step > session.max_step:
                return

    def evaluate(self, model, val_set, msg) -> float:
        model.eval()
        val_loss = 0
        for i, batch in enumerate(val_set, 1):
            stream(msg + f'| Evaluating {i}/{len(val_set)}')
            seqs, mels, stops, ids, lens = batch
            seqs, mels, stops, lens = \
                seqs.to(device), mels.to(device), stops.to(device), lens.to(device)
            with torch.no_grad():
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
        mel_sample = mels.transpose(1, 2)[0, :lens[0]].detach().cpu().numpy()
        gta_sample = post_mels.transpose(1, 2)[0, :lens[0]].detach().cpu().numpy()
        att_sample = att[0].detach().cpu().numpy()
        target_fig = plot_mel(mel_sample)
        gta_fig = plot_mel(gta_sample)
        att_fig = plot_attention(att_sample)
        self.writer.add_figure('Mel/target', target_fig, model.step)
        self.writer.add_figure('Mel/ground_truth_aligned', gta_fig, model.step)
        self.writer.add_figure('Attention/ground_truth_aligned', att_fig, model.step)

        target_wav = self.audio.griffinlim(mel_sample, 32)
        gta_wav = self.audio.griffinlim(gta_sample, 32)
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
        gen_wav = self.audio.griffinlim(gen_sample, 32)
        self.writer.add_audio(
            tag='Wav/generated', snd_tensor=gen_wav,
            global_step=model.step, sample_rate=self.audio.sample_rate)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Entrypoint for training the TacoGan model.')
    parser.add_argument(
        '--config', '-c', help='Point to the config.', default='config.yaml')
    args = parser.parse_args()
    device = get_device()
    paths = Paths()
    latest_ckpt = paths.ckpt/'latest_model.zip'
    cfg = Config.load(args.config)
    if os.path.exists(latest_ckpt):
        print(f'Loading model from {latest_ckpt}')
        model, optimizer, cfg_loaded = load_model(latest_ckpt, device)
        cfg = cfg_loaded.update(cfg)
    else:
        print(f'\nInitialising new model from {args.config}\n')
        model = Tacotron.from_config(cfg).to(device)
        optimizer = optim.Adam(model.parameters())

    trainer = Trainer(cfg)
    trainer.train(model, optimizer)