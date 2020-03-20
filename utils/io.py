import os
import shutil
import sys
import pickle
import ruamel.yaml
from pathlib import Path


def get_files(path: str, extension='.wav'):
    path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))


def stream(msg: str):
    sys.stdout.write(f'\r{msg}')


def progbar(i, total, msg='', size=18):
    done = (i * size) // total
    bar = '█' * done + '░' * (size - done)
    stream(f'{bar} {msg} ')


def pickle_binary(data: object, file: str):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: str):
    with open(file, 'rb') as f:
        return pickle.load(f)


def create_dir(path: str, overwrite=False):
    if overwrite and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

