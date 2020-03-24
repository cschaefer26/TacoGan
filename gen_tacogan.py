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
from text.tokenizer import Tokenizer
from utils.common import Averager
from utils.config import Config
from utils.decorators import ignore_exception
from utils.display import plot_mel, plot_attention, display_params, stream
from utils.paths import Paths


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':

    paths = Paths()
    parser = argparse.ArgumentParser(
        description='Entrypoint for training the TacoGan model.')
    parser.add_argument(
        '--model', '-m', help='Point to the model.zip', default=str(paths.ckpt/'latest_model.zip'))
    parser.add_argument(
        '--text', '-t', help='Input text.')
    args = parser.parse_args()
    device = get_device()
    latest_ckpt = paths.ckpt/'latest_model.zip'
    model, optimizer, cfg = load_model(latest_ckpt, device)
    tokenier = Tokenizer(cfg.symbols)

