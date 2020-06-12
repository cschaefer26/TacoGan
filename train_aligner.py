import argparse

import torch
from torch import optim

from alignment_trainer import AlignmentTrainer
from model.aligner import Aligner
from model.gan import GAN
from model.io import  ModelPackage
from model.tacotron_new import Tacotron
from trainer import Trainer
from utils.config import Config
from utils.io import get_latest_file
from utils.paths import Paths


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
    cfg = Config.load(args.config)
    ckpt_path = paths.ckpt/cfg.config_id
    print(ckpt_path)
    latest_ckpt = get_latest_file(ckpt_path, extension='.zip')

    aligner = Aligner(n_mels=cfg.n_mels, lstm_dim=256, num_symbols=len(cfg.symbols)+1).to(device)
    optimizer = optim.Adam(aligner.parameters())
    trainer = AlignmentTrainer(cfg)
    trainer.train(aligner, optimizer)