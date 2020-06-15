import time

import torch
import torch.nn.functional as F
from torch.nn.functional import log_softmax
from torch.nn.modules.loss import CTCLoss
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio import Audio
from dataset import new_audio_datasets
from losses import MaskedL1
from model.aligner import Aligner
from model.io import ModelPackage
from text.text_cleaner import basic_cleaners
from text.tokenizer import Tokenizer
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


class AlignmentTrainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths = Paths()
        self.audio = Audio(cfg)
        self.ckpt_path = self.paths.ckpt/cfg.config_id
        log_dir = self.ckpt_path/'tensorboard'
        self.writer = SummaryWriter(log_dir=log_dir, comment='v1')
        self.ctc_loss = CTCLoss()
        self.tokenizer = Tokenizer(basic_cleaners, cfg.symbols)


    def train(self, model: Aligner, optimizer: Optimizer):
        for i, session_params in enumerate(self.cfg.training_schedule, 1):
            r, lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = new_audio_datasets(
                    paths=self.paths, batch_size=bs, r=r, cfg=self.cfg)
                session = Session(
                    index=i, r=r, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: Aligner, optimizer: Optimizer, session: Session):
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

            for i, (seqs, mels, seq_lens, mel_lens, ids) in enumerate(session.train_set):
                seqs, mels, seq_lens, mel_lens = \
                    seqs.to(device), mels.to(device), seq_lens.to(device), mel_lens.to(device)
                t_start = time.time()
                block_step = model.get_step() % cfg.steps_to_eval + 1

                model.train()
                pred = model(mels)
                #print(f'pred {pred.shape} seqs {seqs.shape} ml {mel_lens.shape} sl {seq_lens.shape}')

                pred = pred.transpose(0, 1).log_softmax(2)
                loss = self.ctc_loss(pred, seqs, mel_lens, seq_lens)
                loss_avg.add(loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                duration_avg.add(time.time() - t_start)
                steps_per_s = 1. / duration_avg.get()
                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/batch_sze', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                msg = f'{block_step}/{cfg.steps_to_eval} | Step: {model.get_step()} ' \
                      f'| {steps_per_s:#.2} steps/s | Avg. Loss: {loss_avg.get():#.4} '
                stream(msg)

                if model.step % cfg.steps_to_checkpoint == 0:
                    self.save_model(model, optimizer, step=model.get_step())

                first_pred = pred.transpose(0, 1)[0].max(1)[1].detach().cpu().numpy().tolist()
                first_pred_d = self.tokenizer.decode(first_pred)
                first_target = seqs[0].detach().cpu().numpy().tolist()
                first_target_d = self.tokenizer.decode(first_target)
                if model.get_step() % 100 == 0:
                    print()
                    print(f'pred: {first_pred}')
                    print(f'pred dec: {first_pred_d}')
                    print(f'target: {first_target}')
                    print(f'target dec: {first_target_d}')
                    print(first_pred)


                """
                if model.step % self.cfg.steps_to_eval == 0:
                    val_loss = self.evaluate(model, session.val_set, msg)
                    self.writer.add_scalar('Loss/val', val_loss, model.step)
                    self.save_model(model)
                    stream(msg + f'| Val Loss: {float(val_loss):#0.4} \n')
                    loss_avg.reset()
                    duration_avg.reset()
                """
            if model.step > session.max_step:
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
                pred = model(mels)
                lin_mels, post_mels, att = pred
                lin_loss = F.l1_loss(lin_mels, mels)
                post_loss = F.l1_loss(post_mels, mels)
                val_loss += lin_loss + post_loss

        val_loss /= len(val_set)
        return float(val_loss)

    def save_model(self, model: Aligner, optimizer: Optimizer, step=None):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, self.ckpt_path/'latest_model.pyt')
        if step is not None:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, self.ckpt_path/f'model_step_{model.get_step()}.pyt')

