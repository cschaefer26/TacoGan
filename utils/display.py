from typing import List
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_mel(mel: np.array) -> None:
    mel = np.flip(mel, axis=0)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(mel, interpolation='nearest', cmap='plasma', aspect='auto')
    return fig


def plot_attention(attn: np.array) -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    return fig


def stream(msg: str):
    sys.stdout.write(f'\r{msg}')


def progbar(i, total, msg='', size=14):
    done = (i * size) // total
    bar = '█' * done + '░' * (size - done)
    stream(f'{bar} {msg} ')


def display_params(params: List[tuple]):
    it = iter(params)
    if len(params) % 2 != 0:
        params.append(('', ''))
    param_tuples = list(zip(it, it))
    upper_params = []
    lower_params = []
    lines = []
    for p1, p2 in param_tuples:
        p1_key, p1_val = str(p1[0]), str(p1[1])
        p2_key, p2_val = str(p2[0]), str(p2[1])
        p1_key += ': '
        p2_key = p2_key + ': ' if len(p2_key) > 0 else '  '
        p1_width = len(p1_key) + len(p1_val)
        p2_width = len(p2_key) + len(p2_val)
        cell_with = max(p1_width, p2_width)
        p1_pad = ' ' * max(0, cell_with - p1_width)
        p2_pad = ' ' * max(0, cell_with - p2_width)
        p1_str = p1_key + p1_val + p1_pad
        p2_str = p2_key + p2_val + p2_pad
        upper_params.append(' ' + p1_str + ' |')
        lower_params.append(' ' + p2_str + ' |')
        lines.append('-' * len(p1_str) + '--+')
    upper_str = ''.join(upper_params)
    lower_str = ''.join(lower_params)
    lines_str = ''.join(lines)
    print('+-' + lines_str)
    print('| ' + upper_str)
    print('+-' + lines_str)
    print('| ' + lower_str)
    print('+-' + lines_str + '\n')
