import argparse

import torch
from torch import optim

from model.forward_tacotron import ForwardTacotron
from model.gan import GAN
from model.io import  ModelPackage
from model.tacotron_new import Tacotron
from trainer import ForwardTrainer
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
    #latest_ckpt = get_latest_file(ckpt_path, extension='.zip')
    print(f'\nInitialising new model from {args.config}')
    print(f'Checkpoint path: {ckpt_path}')
    forward = Tacotron.from_config(cfg).to(device)
    gan = GAN.from_config(cfg).to(device)
    model = ForwardTacotron.from_config(cfg)
    trainer = ForwardTrainer(cfg)
    trainer.train(model)