import torch
from torch import optim

from dataset import new_audio_datasets
from model.tacotron import Tacotron
from utils.io import read_config
from utils.paths import Paths


def train_session(tacotron, train_set, val_set, max_step):

    while True:
        for i, (seqs, mels, stops, ids, lens) in enumerate(train_set, 1):
            print(f'{i} {ids[0]}')

        if tacotron.step >= max_step:
            return



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cfg = read_config('config.yaml')
    paths = Paths()
    fft_bins = cfg['n_fft'] // 2 + 1

    print('\nInitialising Tacotron Model...\n')
    tacotron = Tacotron(embed_dims=128,
                     num_chars=len(cfg['symbols']),
                     encoder_dims=128,
                     decoder_dims=128,
                     n_mels=cfg['n_mels'],
                     fft_bins=fft_bins,
                     postnet_dims=256,
                     encoder_K=16,
                     lstm_dims=512,
                     postnet_K=8,
                     num_highways=4,
                     dropout=0.5,
                     stop_threshold=-3.4).to(device)

    optimizer = optim.Adam(tacotron.parameters())
    train_set, val_set = new_audio_datasets(paths, 16, 5, cfg)
    device = next(tacotron.parameters()).device

    for schedule in cfg['training_schedule']:
        r, lr, max_step, bs = schedule
        train_set, val_set = new_audio_datasets(
            paths=paths, batch_size=bs, r=5, cfg=cfg)
        print(f'{r} {lr} {max_step} {bs}')
        #train_session(tacotron=tacotron, train_set=train_set, val_set=val_set, max_step=max_step)

