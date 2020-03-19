import matplotlib.pyplot as plt
import numpy as np


def plot_mel(mel, path):
    mel = np.flip(mel, axis=0)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(mel, interpolation='nearest', cmap='plasma', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)


def plot_attention(attn, path):
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)
