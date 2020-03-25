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
from model.io import save_model, load_model, ModelPackage
from model.tacotron_new import Tacotron
from text.text_cleaner import get_cleaners
from text.tokenizer import Tokenizer
from utils.common import Averager
from utils.config import Config
from utils.decorators import ignore_exception
from utils.display import plot_mel, plot_attention, display_params, stream
from utils.io import get_latest_file
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

    if args.model is None:
        model_path = get_latest_file(paths.ckpt, extension='.zip')
        assert model_path is not None, f'No model could be found in {paths.ckpt}'
    else:
        model_path = args.model
        assert model_path is not None, f'No model could be found at {args.model}'

    print(f'Loading model from {model_path}')
    model = ModelPackage.load(model_path, device)
    cleaners = get_cleaners(model.cfg.cleaners)
    tokenier = Tokenizer(cleaners, model.cfg.symbols)
    seq = tokenier.encode(args.text)
    mel, post, att = model.tacotron.generate(seq)
    audio = Audio(model.cfg)
    wav = audio.griffinlim(post)

    display_params([
        ('Model Step', model.get_step()),
        ('Reduction', model.r),
        ('Sample Rate', model.cfg.sample_rate),
        ('Hop Length', model.cfg.hop_length)])

    audio.save_wav(wav, paths.outputs/'sample.wav')
    print(f'model step: {model.get_step()}')
