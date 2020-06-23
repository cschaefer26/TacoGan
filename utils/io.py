import os
import shutil
import sys
import pickle
import ruamel.yaml
import torch
from pathlib import Path

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from utils.config import Config


def save_model(save_path: str, model: Module, opti: Optimizer, cfg: Config):
    torch.save({
        'model': model.state_dict(),
        'optim': opti.state_dict(),
        'config': cfg.to_string()
    }, save_path)


def get_files(path: str, extension='.wav'):
    path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))


def get_latest_file(path: str, extension='.pyt'):
    files = get_files(path, extension=extension)
    if len(files) > 0:
        latest_file = max(files, key=os.path.getctime)
        return latest_file
    else:
        return None


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

