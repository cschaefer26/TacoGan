import argparse
from random import Random
from typing import Tuple

import numpy as np
from pathlib import Path
from audio import Audio
from utils.config import Config
from utils.display import display_params, progbar
from utils.paths import Paths
from utils.io import get_files, pickle_binary
from multiprocessing import Pool, cpu_count


class Preprocessor:

    def __init__(self, audio, mel_path) -> None:
        self.mel_path = mel_path
        self.audio = audio

    def process_wav(self, path: Path) -> Tuple[str, int]:
        mel_id = path.stem
        wav = audio.load_wav(path)
        mel = audio.wav_to_mel(wav)
        np.save(self.mel_path/f'{mel_id}.npy', mel, allow_pickle=False)
        return mel_id, mel.shape[0]


def read_metafile(path: str):
    csv_files = get_files(path, extension='.csv')
    assert len(csv_files) == 1, f'Expected a single csv file, found: {len(csv_files)} '
    text_dict = {}
    with open(csv_files[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[1]
    return text_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocessing script that generates mel spectrograms.')
    parser.add_argument(
        '--path', '-p', help='Point to the data path, expects LJSpeech-like folder.')
    parser.add_argument(
        '--config', '-c', help='Point to the config.', default='config.yaml')
    args = parser.parse_args()
    cfg = Config.load(args.config)

    audio = Audio(cfg)
    paths = Paths()
    preprocessor = Preprocessor(audio, paths.mel)

    files = get_files(args.path)
    n_workers = min(cpu_count()-1, cfg.n_workers)
    pool = Pool(processes=n_workers)
    map_func = pool.imap_unordered(preprocessor.process_wav, files)
    dataset = []

    text_dict = read_metafile(args.path)
    display_params([
        ('Num Train', len(files)-cfg.n_val), ('Num Val', cfg.n_val),
        ('Num Mels', cfg.n_mels), ('Win Length', cfg.win_length),
        ('Hop Length', cfg.hop_length), ('Min Frequency', cfg.fmin),
        ('Sample Rate', cfg.sample_rate), ('CPU Usage', f'{n_workers}/{cpu_count()}'),
    ])
    for i, (mel_id, mel_len) in enumerate(map_func, 1):
        dataset += [(mel_id, mel_len)]
        progbar(i, len(files), f'{i}/{len(files)}')

    dataset = [d for d in dataset if d[0] in text_dict]
    random = Random(cfg.seed)
    random.shuffle(dataset)
    train_dataset = dataset[cfg.n_val:]
    val_dataset = dataset[:cfg.n_val]
    # sort val dataset longest to shortest
    val_dataset.sort(key=lambda d: -d[1])

    pickle_binary(text_dict, paths.data/'text_dict.pkl')
    pickle_binary(train_dataset, paths.data/'train_dataset.pkl')
    pickle_binary(val_dataset, paths.data/'val_dataset.pkl')

    print('done.')

