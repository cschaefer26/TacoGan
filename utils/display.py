import matplotlib.pyplot as plt
import numpy as np


def plot_mel(mel):
    mel = np.flip(mel, axis=0)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(mel, interpolation='nearest', cmap='plasma', aspect='auto')
    return fig


def plot_attention(attn):
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    return fig
