import time

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio import Audio
from dataset import new_audio_datasets
from losses import MaskedL1
from model.io import ModelPackage
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
        self.ckpt_path = self.paths.ckpt/cfg.config_id
        log_dir = self.ckpt_path/'tensorboard'
        self.writer = SummaryWriter(log_dir=log_dir, comment='v1')
        self.criterion = MaskedL1()

    def train(self, model: ModelPackage):
        for i, session_params in enumerate(self.cfg.training_schedule, 1):
            r, lr, max_step, bs = session_params
            if model.tacotron.step < max_step:
                train_set, val_set = new_audio_datasets(
                    paths=self.paths, batch_size=bs, r=r, cfg=self.cfg)
                session = Session(
                    index=i, r=r, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, session)

    def train_session(self, model: ModelPackage, session: Session):
        model.r = session.r
        cfg = self.cfg
        tacotron, gan = model.tacotron, model.gan
        taco_opti, gen_opti, disc_opti = \
            model.taco_opti, model.gen_opti, model.disc_opti
        device = next(tacotron.parameters()).device
        display_params([
            ('Session', session.index), ('Reduction', session.r),
            ('Max Step', session.max_step), ('Learning Rate', session.lr),
            ('Batch Size', session.bs), ('Steps per Epoch', len(session.train_set))
        ])

        for g in taco_opti.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()

        while tacotron.get_step() <= session.max_step:

            for i, (seqs, mels, stops, ids, lens) in enumerate(session.train_set):
                seqs, mels, stops, lens = \
                    seqs.to(device), mels.to(device), stops.to(device), lens.to(device)
                t_start = time.time()
                block_step = tacotron.get_step() % cfg.steps_to_eval + 1

                tacotron.train()
                lin_mels, post_mels, att = tacotron(seqs, mels)

                lin_loss = self.criterion(lin_mels, mels, lens)
                post_loss = self.criterion(post_mels, mels, lens)

                loss = lin_loss + post_loss
                loss_avg.add(loss)

                taco_opti.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tacotron.parameters(), 1.0)
                taco_opti.step()

                duration_avg.add(time.time() - t_start)
                steps_per_s = 1. / duration_avg.get()
                self.writer.add_scalar('Loss/train', loss, tacotron.get_step())
                self.writer.add_scalar('Params/reduction_factor', session.r, tacotron.get_step())
                self.writer.add_scalar('Params/batch_sze', session.bs, tacotron.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, tacotron.get_step())

                msg = f'{block_step}/{cfg.steps_to_eval} | Step: {tacotron.get_step()} ' \
                      f'| {steps_per_s:#.2} steps/s | Avg. Loss: {loss_avg.get():#.4} '
                stream(msg)

                if tacotron.step % cfg.steps_to_checkpoint == 0:
                    self.save_model(model, step=tacotron.get_step())

                if tacotron.step % self.cfg.steps_to_eval == 0:
                    val_loss = self.evaluate(model, session.val_set, msg)
                    self.writer.add_scalar('Loss/val', val_loss, tacotron.step)
                    self.save_model(model)
                    stream(msg + f'| Val Loss: {float(val_loss):#0.4} \n')
                    loss_avg.reset()
                    duration_avg.reset()

            if tacotron.step > session.max_step:
                return

    def evaluate(self, model, val_set, msg) -> float:
        model.tacotron.eval()
        val_loss = 0
        device = next(model.tacotron.parameters()).device
        for i, batch in enumerate(val_set, 1):
            stream(msg + f'| Evaluating {i}/{len(val_set)}')
            seqs, mels, stops, ids, lens = batch
            seqs, mels, stops, lens = \
                seqs.to(device), mels.to(device), stops.to(device), lens.to(device)
            with torch.no_grad():
                pred = model.tacotron(seqs, mels)
                lin_mels, post_mels, att = pred
                lin_loss = F.l1_loss(lin_mels, mels)
                post_loss = F.l1_loss(post_mels, mels)
                val_loss += lin_loss + post_loss
            if i == 1:
                self.generate_samples(model, batch, pred)

        val_loss /= len(val_set)
        return float(val_loss)

    def save_model(self, model: ModelPackage, step=None):
        model.save(self.ckpt_path/'latest_model.zip')
        if step is not None:
            model.save(self.ckpt_path/f'model_step_{step}.zip')

    @ignore_exception
    def generate_samples(self, model: ModelPackage,
                         batch: torch.Tensor, pred: torch.Tensor):
        seqs, mels, stops, ids, lens = batch
        lin_mels, post_mels, att = pred
        mel_sample = mels.transpose(1, 2)[0, :lens[0]].detach().cpu().numpy()
        gta_sample = post_mels.transpose(1, 2)[0, :lens[0]].detach().cpu().numpy()
        att_sample = att[0].detach().cpu().numpy()
        target_fig = plot_mel(mel_sample)
        gta_fig = plot_mel(gta_sample)
        att_fig = plot_attention(att_sample)
        self.writer.add_figure('Mel/target', target_fig, model.tacotron.step)
        self.writer.add_figure('Mel/ground_truth_aligned', gta_fig, model.tacotron.step)
        self.writer.add_figure('Attention/ground_truth_aligned', att_fig, model.tacotron.step)

        target_wav = self.audio.griffinlim(mel_sample, 32)
        gta_wav = self.audio.griffinlim(gta_sample, 32)
        self.writer.add_audio(
            tag='Wav/target', snd_tensor=target_wav,
            global_step=model.tacotron.step, sample_rate=self.audio.sample_rate)
        self.writer.add_audio(
            tag='Wav/ground_truth_aligned', snd_tensor=gta_wav,
            global_step=model.tacotron.step, sample_rate=self.audio.sample_rate)

        seq = seqs[0].tolist()
        _, gen_sample, att_sample = model.tacotron.generate(seq, steps=lens[0])
        gen_fig = plot_mel(gen_sample)
        att_fig = plot_attention(att_sample)
        self.writer.add_figure('Attention/generated', att_fig, model.tacotron.step)
        self.writer.add_figure('Mel/generated', gen_fig, model.tacotron.step)
        gen_wav = self.audio.griffinlim(gen_sample, 32)
        self.writer.add_audio(
            tag='Wav/generated', snd_tensor=gen_wav,
            global_step=model.tacotron.step, sample_rate=self.audio.sample_rate)

