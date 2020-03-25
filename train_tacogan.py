import argparse

import torch
from torch import optim

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
    if latest_ckpt:
        print(f'\nLoading model from {latest_ckpt}')
        model = ModelPackage.load(latest_ckpt, device)
        model.cfg.update(cfg)
    else:
        print(f'\nInitialising new model from {args.config}')
        print(f'Checkpoint path: {ckpt_path}')
        tacotron = Tacotron.from_config(cfg).to(device)
        gan = GAN.from_config(cfg).to(device)
        taco_opti = optim.Adam(tacotron.parameters())
        gen_opti = optim.Adam(gan.generator.parameters())
        disc_opti = optim.Adam(gan.discriminator.parameters())
        model = ModelPackage(
            tacotron=tacotron, gan=gan, taco_opti=taco_opti,
            gen_opti=gen_opti, disc_opti=disc_opti, cfg=cfg)

    trainer = Trainer(model.cfg)
    trainer.train(model)