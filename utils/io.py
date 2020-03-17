import sys
import pickle
import ruamel.yaml
from pathlib import Path


def read_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)


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
