from typing import List, Dict
import torch
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from text.text_cleaner import english_cleaners, german_cleaners
from text.tokenizer import Tokenizer
from utils.io import unpickle_binary
from utils.paths import Paths


class AudioDataset(Dataset):

    def __init__(self,
                 mel_path: Path,
                 mel_ids: List[str],
                 text_dict: Dict[str, str],
                 tokenizer: Tokenizer):
        super().__init__()
        self.mel_path = mel_path
        self.mel_ids = mel_ids
        self.text_dict = text_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        mel_id = self.mel_ids[index]
        text = self.text_dict[mel_id]
        seq = self.tokenizer.encode(text)
        mel = np.load(str(self.mel_path/f'{mel_id}.npy'))
        mel_len = mel.shape[0]
        return seq, mel, mel_id, mel_len

    def __len__(self):
        return len(self.mel_ids)


def new_audio_datasets(paths: Paths, batch_size, r, cfg):
    train_path = str(paths.data/'train_dataset.pkl')
    val_path = str(paths.data/'val_dataset.pkl')
    train_dataset = unpickle_binary(train_path)
    val_dataset = unpickle_binary(val_path)
    train_dataset = [d for d in train_dataset if d[1] <= cfg.max_mel_len]
    val_dataset = [d for d in val_dataset if d[1] <= cfg.max_mel_len]
    train_ids, train_lens = zip(*train_dataset)
    val_ids, val_lens = zip(*val_dataset)
    text_path = str(paths.data/'text_dict.pkl')
    text_dict = unpickle_binary(text_path)

    if cfg.cleaners == 'english_cleaners':
        cleaners = english_cleaners
    elif cfg.cleaners == 'german_cleaners':
        cleaners = german_cleaners
    else:
        cl = cfg.cleaners
        raise ValueError(f'cleaners not supported: {cl}')

    tokenizer = Tokenizer(cleaners, cfg.symbols)

    train_dataset = AudioDataset(mel_path=paths.mel,
                                 mel_ids=train_ids,
                                 text_dict=text_dict,
                                 tokenizer=tokenizer)

    val_dataset = AudioDataset(mel_path=paths.mel,
                               mel_ids=val_ids,
                               text_dict=text_dict,
                               tokenizer=tokenizer)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_fn(batch, r),
                           batch_size=batch_size,
                           sampler=None,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=lambda batch: collate_fn(batch, r),
                         batch_size=batch_size,
                         sampler=None,
                         shuffle=False,
                         num_workers=1,
                         drop_last=False,
                         pin_memory=True)

    return train_set, val_set


def collate_fn(batch: tuple, r: int) -> tuple:
    seqs, mels, ids, mel_lens = zip(*batch)
    seq_lens = [len(seq) for seq in seqs]
    max_seq_len = max(seq_lens)
    stops = [_new_stops(seq) for seq in seqs]
    max_mel_len = max(mel_lens)
    if max_mel_len % r != 0:
        max_mel_len += r - max_mel_len % r
    seqs = _to_tensor_1d(seqs, max_seq_len)
    stops = _to_tensor_1d(stops, max_seq_len)
    mels = _to_tensor_2d(mels, max_mel_len)
    mel_lens = torch.tensor(mel_lens)
    return seqs, mels, stops, ids, mel_lens


def _new_stops(seq):
    stops = np.zeros((len(seq)))
    stops[-1] = 1
    return stops


def _to_tensor_1d(seqs: List[np.array], max_len: int):
    seqs_padded = []
    for seq in seqs:
        seq = np.pad(seq, (0, max_len - len(seq)), mode='constant')
        seqs_padded.append(seq)
    seqs_padded = np.stack(seqs_padded)
    return torch.tensor(seqs_padded, dtype=torch.long)


def _to_tensor_2d(arrs: List[np.array], max_len: int):
    arrs_padded = []
    for arr in arrs:
        arr = np.pad(arr, ((0, max_len - arr.shape[0]), (0, 0)), mode='constant')
        arrs_padded.append(arr)
    arrs_padded = np.stack(arrs_padded)
    return torch.tensor(arrs_padded, dtype=torch.float32)
