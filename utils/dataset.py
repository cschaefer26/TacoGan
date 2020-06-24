import random
from typing import List, Dict
import torch
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from text.tokenizer import Tokenizer
from utils.io import unpickle_binary
from utils.paths import Paths


class AudioDataset(Dataset):

    def __init__(self,
                 mel_path: Path,
                 dur_path: Path,
                 mel_ids: List[str],
                 text_dict: Dict[str, str],
                 tokenizer: Tokenizer):
        super().__init__()
        self.mel_path = mel_path
        self.dur_path = dur_path
        self.mel_ids = mel_ids
        self.text_dict = text_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        mel_id = self.mel_ids[index]
        text = self.text_dict[mel_id]
        seq = self.tokenizer.encode(text)
        mel = np.load(str(self.mel_path/f'{mel_id}.npy'))
        dur = np.load(str(self.dur_path/f'{mel_id}.npy'))
        mel_len = mel.shape[0]
        seq_len = len(seq)
        return seq, mel, dur, seq_len, mel_len, mel_id

    def __len__(self):
        return len(self.mel_ids)


class AlignerDataset(Dataset):

    def __init__(self,
                 mel_path: Path,
                 mel_ids: List[str],
                 text_dict,
                 tokenizer):
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
        seq_len = len(seq)
        return seq, mel, seq_len, mel_len, mel_id

    def __len__(self):
        return len(self.mel_ids)


# from https://github.com/fatchord/WaveRNN/blob/master/utils/dataset.py
class BinnedLengthSampler(Sampler):

    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []
        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]
        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)
        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])
        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


def new_aligner_dataset(paths: Paths, batch_size, cfg):
    train_path = str(paths.data/'train_dataset.pkl')
    val_path = str(paths.data/'val_dataset.pkl')
    train_dataset = unpickle_binary(train_path)
    val_dataset = unpickle_binary(val_path)
    comb_dataset = val_dataset + train_dataset
    mel_ids, mel_lens = zip(*comb_dataset)
    text_path = str(paths.data/'text_dict.pkl')
    text_dict = unpickle_binary(text_path)
    tokenizer = Tokenizer(cfg.symbols)
    train_sampler = BinnedLengthSampler(mel_lens, batch_size, batch_size * 3)

    train_dataset = AlignerDataset(mel_path=paths.mel,
                                   mel_ids=mel_ids,
                                   text_dict=text_dict,
                                   tokenizer=tokenizer)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_aligner(batch),
                           batch_size=batch_size,
                           sampler=train_sampler,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True)
    return train_set


def new_audio_datasets(paths: Paths, batch_size, cfg):
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
    train_sampler = BinnedLengthSampler(train_lens, batch_size, batch_size * 3)

    tokenizer = Tokenizer(cfg.symbols)

    train_dataset = AudioDataset(mel_path=paths.mel,
                                 dur_path=paths.dur,
                                 mel_ids=train_ids,
                                 text_dict=text_dict,
                                 tokenizer=tokenizer)

    val_dataset = AudioDataset(mel_path=paths.mel,
                               dur_path=paths.dur,
                               mel_ids=val_ids,
                               text_dict=text_dict,
                               tokenizer=tokenizer)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_forward(batch),
                           batch_size=batch_size,
                           sampler=train_sampler,
                           num_workers=1,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=lambda batch: collate_forward(batch),
                         batch_size=batch_size,
                         sampler=None,
                         shuffle=False,
                         num_workers=1,
                         drop_last=False,
                         pin_memory=True)

    return train_set, val_set


def collate_aligner(batch: tuple) -> tuple:
    seqs, mels, seq_lens, mel_lens, ids = zip(*batch)
    seq_lens = [len(seq) for seq in seqs]
    max_seq_len = max(seq_lens)
    max_mel_len = max(mel_lens)
    seqs = _to_tensor_1d(seqs, max_seq_len)
    mels = _to_tensor_2d(mels, max_mel_len)
    seq_lens = torch.tensor(seq_lens)
    mel_lens = torch.tensor(mel_lens)
    return seqs, mels, seq_lens, mel_lens, ids


def collate_forward(batch: tuple) -> tuple:
    seqs, mels, durs, seq_lens, mel_lens, ids = zip(*batch)
    seq_lens = [len(seq) for seq in seqs]
    max_seq_len = max(seq_lens)
    max_mel_len = max(mel_lens)
    seqs = _to_tensor_1d(seqs, max_seq_len)
    mels = _to_tensor_2d(mels, max_mel_len)
    seq_lens = torch.tensor(seq_lens)
    mel_lens = torch.tensor(mel_lens)
    return seqs, mels, seq_lens, mel_lens, ids


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
        arr = np.pad(arr, ((0, max_len - arr.shape[0]), (0, 0)),
                     constant_values=0, mode='constant')
        arrs_padded.append(arr)
    arrs_padded = np.stack(arrs_padded)
    return torch.tensor(arrs_padded, dtype=torch.float32)
