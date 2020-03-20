import os
import shutil
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.adam import Adam

from model.tacotron import Tacotron
from utils.io import create_dir, save_config, load_config


def save_model(model: Tacotron, optimizer: Adam, cfg: dict, path: Path) -> None:
    assert str(path).endswith('.zip'), 'Model path should end with .zip!'
    path = path.parent/path.stem
    tmp_dir = Path(str(path) + '_tmp_save')
    create_dir(tmp_dir, overwrite=True)
    torch.save(model.state_dict(), tmp_dir/'model.pyt')
    torch.save(optimizer.state_dict(), tmp_dir/'optimizer.pyt')
    save_config(cfg, tmp_dir/'config.yaml')
    shutil.make_archive(path, 'zip', tmp_dir)
    shutil.rmtree(tmp_dir)


def load_model(path: Path, device='cpu') -> tuple:
    device = torch.device(device)
    tmp_dir = Path(str(path) + '_tmp_load')
    shutil.unpack_archive(str(path), extract_dir=tmp_dir)
    cfg = load_config(tmp_dir/'config.yaml')

    model = Tacotron.from_config(cfg).to(device)
    state_dict = torch.load(tmp_dir/'model.pyt', device)
    model.load_state_dict(state_dict, strict=False)

    optimizer = Adam(model.parameters())
    state_dict = torch.load(tmp_dir/'optimizer.pyt', device)
    optimizer.load_state_dict(state_dict)
    shutil.rmtree(tmp_dir)
    return model, optimizer, cfg



