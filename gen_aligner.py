import argparse

import torch

from model.aligner import Aligner
from preprocessing.audio import Audio
from text.tokenizer import Tokenizer
from utils.config import Config
from utils.display import display_params
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
        '--model', '-m', help='Point to the model .pyt file')
    parser.add_argument(
        '--text', '-t', help='Input text.')
    args = parser.parse_args()
    device = get_device()

    print(f'Loading model from {args.model}')

    checkpoint = torch.load(args.model, map_location=device)

    cfg = Config.from_string(checkpoint['config'])
    model = Aligner.from_config(cfg)

    print(f'Loaded aligner with step {model.get_step()}')
