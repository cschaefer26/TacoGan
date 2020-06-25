import librosa
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph._shortest_path import dijkstra
from scipy.sparse.csr import csr_matrix
import torch
import argparse

from model.aligner import Aligner
from model.forward_tacotron import ForwardTacotron
from preprocessing.audio import Audio
from text.text_cleaner import get_cleaner
from text.tokenizer import Tokenizer
from utils.config import Config
from utils.dataset import new_aligner_dataset
from utils.io import unpickle_binary
from utils.paths import Paths


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Entrypoint for training the TacoGan model.')
    parser.add_argument(
        '--model', '-m', help='Point to the model pyt.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    checkpoint = torch.load(args.model, map_location=device)

    cfg = Config.from_string(checkpoint['config'])
    model = ForwardTacotron.from_config(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    paths = Paths()
    audio = Audio(cfg)
    print(f'loaded forward step {model.get_step()}')
    text_dict = unpickle_binary('data/text_dict.pkl')

    #text = 'President Trump met with other leaders at the group of 20 conference.'

    print(text_dict)
    cleaner = get_cleaner(cfg.language)
    tokenizer = Tokenizer(cfg.symbols)
    print(cfg.language)
    text = cleaner(text)
    print(text)
    seq = tokenizer.encode(text)
    _, mel, dur = model.generate(seq)

    wav = audio.griffinlim(mel)
    librosa.output.write_wav('/tmp/sample.wav', wav.astype(np.float32), sr=audio.sample_rate)


