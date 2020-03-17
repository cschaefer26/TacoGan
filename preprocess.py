# Adapted from https://github.com/fatchord/WaveRNN
from typing import Tuple

import numpy as np
from pathlib import Path
from audio import Audio
from paths import Paths
from utils import read_config, get_files, progbar, pickle_binary
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
        return mel_id, mel.shape[-1]


def read_metafile(path: str):
    csv_files = get_files(path, extension='.csv')
    assert len(csv_files) == 1, f'Expected a single csv file, found: {len(csv_files)} '
    text_dict = {}
    with open(csv_files[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[-1]
    return text_dict


if __name__ == '__main__':

    path = '/Users/cschaefe/datasets/LJSpeech/LJSpeech-1.1'
    cfg = read_config('config.yaml')
    audio = Audio(cfg)
    paths = Paths()
    preprocessor = Preprocessor(audio, paths.mel)

    files = get_files(path)
    n_workers = min(cpu_count(), cfg['n_workers'])
    pool = Pool(processes=n_workers)
    map_func = pool.imap_unordered(preprocessor.process_wav, files)
    dataset = []

    text_dict = read_metafile(path)

    for i, (mel_id, mel_len) in enumerate(map_func, 1):
        dataset += [(mel_id, mel_len)]
        progbar(i, len(files), f'{i}/{len(files)}')

    # filter ids that are not present in the text
    dataset = [d for d in dataset if d[0] in text_dict]

    pickle_binary(text_dict, paths.data/'text_dict.pkl')
    pickle_binary(dataset, paths.data/'dataset.pkl')


