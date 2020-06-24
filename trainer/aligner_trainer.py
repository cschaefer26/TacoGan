import time

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CTCLoss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.aligner import Aligner
from preprocessing.audio import Audio
from utils.dataset import new_audio_datasets, new_aligner_dataset
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


class AlignerTrainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths = Paths()
        self.audio = Audio(cfg)
        self.ckpt_path = self.paths.ckpt/cfg.config_id
        log_dir = self.ckpt_path/'tensorboard'
        self.writer = SummaryWriter(log_dir=log_dir, comment='v1')
        self.criterion = CTCLoss()

    def train(self, model: Aligner, opti: Optimizer):
        for i, session_params in enumerate(self.cfg.aligner_training_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set = new_aligner_dataset(
                    paths=self.paths, batch_size=bs, cfg=self.cfg)
                session = Session(
                    index=i, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=None)
                self.train_session(model, opti, session)

    def train_session(self, model: ForwardTacotron, opti: Optimizer, session: Session):
        cfg = self.cfg
        device = next(model.parameters()).device
        display_params([
            ('Session', session.index), ('Max Step', session.max_step),
            ('Learning Rate', session.lr), ('Batch Size', session.bs),
            ('Steps per Epoch', len(session.train_set))
        ])

        for g in opti.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()

        while model.get_step() <= session.max_step:

            for i, (seqs, mels, seq_lens, mel_lens, ids) in enumerate(session.train_set):
                seqs, mels, seq_lens, mel_lens = \
                    seqs.to(device), mels.to(device), seq_lens.to(device), mel_lens.to(device)
                t_start = time.time()

                model.train()
                pred = model(mels)
                pred = pred.transpose(0, 1).log_softmax(2)
                loss = self.criterion(pred, seqs, mel_lens, seq_lens)
                loss_avg.add(loss)

                opti.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opti.step()

                duration_avg.add(time.time() - t_start)
                steps_per_s = 1. / duration_avg.get()
                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/batch_sze', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                msg = f'Step: {model.get_step()} ' \
                      f'| {steps_per_s:#.2} steps/s | Avg. Loss: {loss_avg.get():#.4} '
                stream(msg)

                if model.get_step() % cfg.aligner_steps_to_checkpoint == 0:
                    self.save_model(model, opti, step=model.get_step())

                loss_avg.reset()

            if model.get_step() > session.max_step:
                return

    def save_model(self, model: Aligner, opti: Optimizer, step=None):
        save_model(self.ckpt_path/f'latest_model.pyt', model, opti, self.cfg)
        if step is not None:
            save_model(self.ckpt_path / f'model_step{step}.pyt', model, opti, self.cfg)
