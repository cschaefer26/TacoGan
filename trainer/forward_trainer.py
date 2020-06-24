import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from preprocessing.audio import Audio
from utils.dataset import new_audio_datasets
from utils.io import save_model
from utils.losses import MaskedL1
from model.forward_tacotron import ForwardTacotron
from utils.common import Averager
from utils.config import Config
from utils.decorators import ignore_exception
from utils.display import plot_mel, plot_attention, display_params, stream
from utils.paths import Paths


class Session:

    def __init__(self,
                 index: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: DataLoader = None) -> None:
        self.index = index
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set


class ForwardTrainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths = Paths()
        self.audio = Audio(cfg)
        self.ckpt_path = self.paths.ckpt/cfg.config_id
        log_dir = self.ckpt_path/'tensorboard'
        self.writer = SummaryWriter(log_dir=log_dir, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model: ForwardTacotron, opti: Optimizer):
        for i, session_params in enumerate(self.cfg.forward_training_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = new_audio_datasets(
                    paths=self.paths, batch_size=bs, cfg=self.cfg)
                session = Session(
                    index=i, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, opti, session)

    def train_session(self, model: ForwardTacotron, opti: Optimizer, session: Session):
        cfg = self.cfg
        device = next(model.parameters()).device
        display_params([
            ('Session', session.index),
            ('Max Step', session.max_step), ('Learning Rate', session.lr),
            ('Batch Size', session.bs), ('Steps per Epoch', len(session.train_set))
        ])

        for g in opti.param_groups:
            g['lr'] = session.lr

        mel_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()

        while model.get_step() <= session.max_step:

            for i, (seqs, mels, durs, seq_lens, mel_lens, ids) in enumerate(session.train_set):
                seqs, mels, durs, seq_lens, mel_lens = \
                    seqs.to(device), mels.to(device), durs.to(device), seq_lens.to(device), mel_lens.to(device)
                t_start = time.time()

                model.train()
                m1_hat, m2_hat, dur_hat = model(seqs, mels, durs)

                m1_loss = self.l1_loss(m1_hat, mels, mel_lens)
                m2_loss = self.l1_loss(m2_hat, mels, mel_lens)

                dur_loss = self.l1_loss(dur_hat.unsqueeze(-1), durs.unsqueeze(-1), seq_lens)

                loss = m1_loss + m2_loss + dur_loss
                opti.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opti.step()
                mel_loss_avg.add(m1_loss.item() + m2_loss.item())
                dur_loss_avg.add(dur_loss.item())

                duration_avg.add(time.time() - t_start)
                steps_per_s = 1. / duration_avg.get()
                self.writer.add_scalar('Mel_Loss/train', loss, model.get_step())
                self.writer.add_scalar('Dur_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_sze', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                msg = f'Step: {model.get_step()} ' \
                      f'| {steps_per_s:#.2} steps/s | Avg. Mel Loss: {mel_loss_avg.get():#.4} ' \
                      f'| Avg. Duration Loss: {dur_loss_avg.get():#.4} '
                stream(msg)

                if model.get_step() % cfg.forward_steps_to_checkpoint == 0:
                    self.save_model(model, opti, step=model.get_step())

                if model.get_step() % cfg.forward_steps_to_plot == 0:
                    self.generate_plots(model, session.val_set)

                if model.get_step() % self.cfg.forward_steps_to_eval == 0:
                    val_mel_loss, val_dur_loss = self.evaluate(model, session.val_set)
                    self.writer.add_scalar('Mel_Loss/val', val_mel_loss, model.get_step())
                    self.writer.add_scalar('Dur_Loss/val', val_mel_loss, model.get_step())
                    self.save_model(model, opti)
                    stream(msg + f'| Val Loss: {float(val_mel_loss):#0.4} \n')

            mel_loss_avg.reset()
            dur_loss_avg.reset()
            duration_avg.reset()

            if model.get_step() > session.max_step:
                return

    def evaluate(self, model: ForwardTacotron, val_set: DataLoader) -> Tuple[float, float]:
        model.eval()
        m_val_loss = 0
        dur_val_loss = 0
        device = next(model.parameters()).device
        for i, (seqs, mels, durs, seq_lens, mel_lens, ids) in enumerate(val_set, 1):
            seqs, mels, seq_lens, mel_lens = \
                seqs.to(device), mels.to(device), seq_lens.to(device), mel_lens.to(device)
            with torch.no_grad():
                mels = mels.transpose(1, 2)
                m1_hat, m2_hat, dur_hat = model(seqs, mels, durs)
                mels = mels.transpose(1, 2)
                m1_hat = m1_hat.transpose(1, 2)
                m2_hat = m2_hat.transpose(1, 2)
                m1_loss = self.l1_loss(m1_hat, mels, mel_lens)
                m2_loss = self.l1_loss(m2_hat, mels, mel_lens)
                dur_loss = self.l1_loss(dur_hat.unsqueeze(-1), durs.unsqueeze(-1), seq_lens)
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
        return m_val_loss / len(val_set), dur_val_loss / len(val_set)

    def save_model(self, model: ForwardTacotron, opti: Optimizer, step=None):
        save_model(self.ckpt_path/f'latest_forward.pyt', model, opti, self.cfg)
        if step is not None:
            save_model(self.ckpt_path / f'forward_step{step}.pyt', model, opti, self.cfg)

    @ignore_exception
    def generate_plots(self, model: ForwardTacotron, val_set: DataLoader):
        batch = next(iter(val_set))
        seqs, mels, durs, seq_lens, mel_lens, ids = batch
        m1_hat, m2_hat, dur_hat = model(seqs, mels, durs)
        mel_sample = mels.transpose(1, 2)[0, :mel_lens[0]].detach().cpu().numpy()
        gta_sample = m2_hat.transpose(1, 2)[0, :mel_lens[0]].detach().cpu().numpy()
        target_fig = plot_mel(mel_sample)
        gta_fig = plot_mel(gta_sample)
        self.writer.add_figure('Mel/target', target_fig, model.get_step())
        self.writer.add_figure('Mel/ground_truth_aligned', gta_fig, model.get_step())

        target_wav = self.audio.griffinlim(mel_sample, 32)
        gta_wav = self.audio.griffinlim(gta_sample, 32)
        self.writer.add_audio(
            tag='Wav/target', snd_tensor=target_wav,
            global_step=model.get_step(), sample_rate=self.audio.sample_rate)
        self.writer.add_audio(
            tag='Wav/ground_truth_aligned', snd_tensor=gta_wav,
            global_step=model.get_step(), sample_rate=self.audio.sample_rate)

        seq = seqs[0].tolist()
        _, gen_sample, _ = model.generate(seq)
        gen_fig = plot_mel(gen_sample)
        self.writer.add_figure('Mel/generated', gen_fig, model.get_step())
        gen_wav = self.audio.griffinlim(gen_sample, 32)
        self.writer.add_audio(
            tag='Wav/generated', snd_tensor=gen_wav,
            global_step=model.get_step(), sample_rate=self.audio.sample_rate)

