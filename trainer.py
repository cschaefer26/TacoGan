import time

import torch
import torch.nn.functional as F
from torch.nn import BCELoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


from audio import Audio
from dataset import new_audio_datasets
from losses import MaskedL1, MaskedBCE
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
        self.taco_loss = MaskedL1()
        self.gen_loss = MaskedL1()
        self.disc_loss = MaskedBCE()

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
        cfg = self.cfg
        tacotron, gan, generator, discriminator = \
            model.tacotron, model.gan, model.gan.generator, model.gan.discriminator
        taco_opti, gen_opti, disc_opti = \
            model.taco_opti, model.gen_opti, model.disc_opti
        tacotron.r = session.r
        device = next(tacotron.parameters()).device
        display_params([
            ('Session', session.index), ('Reduction', session.r),
            ('Max Step', session.max_step), ('Learning Rate', session.lr),
            ('Batch Size', session.bs), ('Steps per Epoch', len(session.train_set))
        ])

        for g in taco_opti.param_groups:
            g['lr'] = session.lr
        for g in gen_opti.param_groups:
            g['lr'] = session.lr
        for g in disc_opti.param_groups:
            g['lr'] = session.lr

        taco_loss_avg = Averager()
        post_loss_avg = Averager()
        gen_loss_avg = Averager()
        gen_loss_l1_avg = Averager()
        disc_loss_real_avg = Averager()
        disc_loss_fake_avg = Averager()
        duration_avg = Averager()

        while tacotron.get_step() <= session.max_step:

            for i, (seqs, mels, stops, ids, lens) in enumerate(session.train_set):
                seqs, mels, stops, lens = \
                    seqs.to(device), mels.to(device), stops.to(device), lens.to(device)
                fake = torch.zeros((mels.size(0), mels.size(1))).to(device)
                real = torch.ones((mels.size(0), mels.size(1))).to(device)
                t_start = time.time()

                # train tacotron
                tacotron.train()
                gan.train()
                lin_mels, post_mels, att = tacotron(seqs, mels)
                gan.zero_grad()
                tacotron.zero_grad()
                taco_opti.zero_grad()
                lin_loss = self.taco_loss(lin_mels, mels, lens)
                post_loss = self.taco_loss(post_mels, mels, lens)
                loss = lin_loss + post_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tacotron.parameters(), 1.0)
                taco_opti.step()
                taco_loss_avg.add(loss)
                post_loss_avg.add(post_loss)

                # train discriminator
                post_mels = post_mels.detach()
                gan.zero_grad()
                disc_opti.zero_grad()
                with torch.no_grad():
                    gan_mels = generator(post_mels)
                d_fake = discriminator(gan_mels).squeeze()
                d_real = discriminator(mels).squeeze()
                d_loss_fake = self.disc_loss(d_fake, fake, lens)
                d_loss_real = self.disc_loss(d_real, real, lens)
                d_loss = d_loss_fake + d_loss_real
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                disc_opti.step()
                disc_loss_real_avg.add(d_loss_real)
                disc_loss_fake_avg.add(d_loss_fake)

                # train generator
                gan.zero_grad()
                gen_opti.zero_grad()
                gan_mels = generator(post_mels)
                g_l1_loss = self.gen_loss(gan_mels, mels, lens)
                d_fake = discriminator(gan_mels).squeeze()
                d_loss_fake_real = self.disc_loss(d_fake, real, lens)
                g_loss = g_l1_loss + cfg.gan_weight * d_loss_fake_real
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                gen_opti.step()
                gen_loss_avg.add(d_loss_fake_real)
                gen_loss_l1_avg.add(g_l1_loss)

                duration_avg.add(time.time() - t_start)
                steps_per_s = 1. / duration_avg.get()
                self.writer.add_scalar('Loss/train_taco', loss, tacotron.get_step())
                self.writer.add_scalar('Loss/train_post', post_loss, tacotron.get_step())
                self.writer.add_scalar('Loss/train_generator_l1', g_l1_loss, tacotron.get_step())
                self.writer.add_scalar('Loss/train_generator_gan', d_loss_fake_real, tacotron.get_step())
                self.writer.add_scalar('Loss/train_disc_real', d_loss_real, tacotron.get_step())
                self.writer.add_scalar('Loss/train_disc_fake', d_loss_fake, tacotron.get_step())
                self.writer.add_scalar('Params/reduction_factor', session.r, tacotron.get_step())
                self.writer.add_scalar('Params/batch_sze', session.bs, tacotron.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, tacotron.get_step())

                msg = f'Step: {tacotron.get_step()} ' \
                      f'| {steps_per_s:#.2} steps/s | Taco Loss: {taco_loss_avg.get():#.4} ' \
                      f'| Post Loss: {post_loss_avg.get():#.4}  Gen L1 Loss: {gen_loss_l1_avg.get():#.4} ' \
                      f'| Gen GAN Loss: {gen_loss_avg.get()} | Disc Loss Fake: {disc_loss_fake_avg.get():#.4} ' \
                      f'Disc Loss Real: {disc_loss_real_avg.get():#.4} '
                stream(msg)

                if tacotron.step % cfg.steps_to_checkpoint == 0:
                    self.save_model(model, step=tacotron.get_step())

                if tacotron.step % self.cfg.steps_to_eval == 0:
                    val_loss = self.evaluate(model, session.val_set, msg)
                    self.writer.add_scalar('Loss/val', val_loss, tacotron.step)
                    self.save_model(model)
                    stream(msg + f'| Val Loss: {float(val_loss):#0.4} \n')
                    taco_loss_avg.reset()
                    duration_avg.reset()
                    gen_loss_avg.reset()
                    gen_loss_l1_avg.reset()
                    disc_loss_fake_avg.reset()
                    disc_loss_real_avg.reset()
                    taco_loss_avg.reset()
                    post_loss_avg.reset()

            if tacotron.step > session.max_step:
                return

    def evaluate(self, model, val_set, msg) -> float:
        model.tacotron.eval()
        model.gan.eval()
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
        device = next(model.tacotron.parameters()).device
        lin_mels, post_mels, att = pred
        mel_sample = mels.transpose(1, 2)[0, :600].detach().cpu().numpy()
        gta_sample = post_mels.transpose(1, 2)[0, :600].detach().cpu().numpy()
        gan_sample = model.gan.generator(post_mels).transpose(1, 2)[0, :600].detach().cpu().numpy()

        att_sample = att[0].detach().cpu().numpy()
        target_fig = plot_mel(mel_sample)
        gta_fig = plot_mel(gta_sample)
        gan_fig = plot_mel(gan_sample)
        att_fig = plot_attention(att_sample)
        self.writer.add_figure('Mel/target', target_fig, model.tacotron.step)
        self.writer.add_figure('Mel/ground_truth_aligned', gta_fig, model.tacotron.step)
        self.writer.add_figure('Mel/ground_truth_aligned_gan', gan_fig, model.tacotron.step)
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
        _, gen_sample_in, att_sample = model.tacotron.generate(seq, steps=lens[0], batch=True)
        gan_sample = model.gan.generator(gen_sample_in).transpose(1, 2)[0, :600].detach().cpu().numpy()
        gen_fig = plot_mel(gen_sample)
        gan_fig = plot_mel(gan_sample)
        att_fig = plot_attention(att_sample)
        self.writer.add_figure('Attention/generated', att_fig, model.tacotron.step)
        self.writer.add_figure('Mel/generated', gen_fig, model.tacotron.step)
        self.writer.add_figure('Mel/generated_gan', gan_fig, model.tacotron.step)
        gen_wav = self.audio.griffinlim(gen_sample, 32)
        gan_wav = self.audio.griffinlim(gan_sample, 32)
        self.writer.add_audio(
            tag='Wav/generated', snd_tensor=gen_wav,
            global_step=model.tacotron.step, sample_rate=self.audio.sample_rate)
        self.writer.add_audio(
            tag='Wav/generated_gan', snd_tensor=gan_wav,
            global_step=model.tacotron.step, sample_rate=self.audio.sample_rate)
