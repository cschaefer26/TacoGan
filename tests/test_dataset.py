import numpy as np
import os
import unittest
from pathlib import Path
from audio import Audio
from dataset import AudioDataset, collate_fn
from text.tokenizer import Tokenizer
from utils.io import read_config


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = Path(current_dir)
        config_path = current_dir/'test_config.yaml'
        self.mel_path = current_dir/'resources'
        self.cfg = read_config(config_path)
        self.audio = Audio(self.cfg)

    def test_audio_dataset(self):
        mel_ids = ['small_sample']
        text_dict = {'small_sample': 'Small sample text.'}
        cleaner = lambda x: x.lower()
        symbols = 'abcdefghijklmnopqrstuvwxyz. '
        tokenizer = Tokenizer(cleaner=cleaner, symbols=symbols)
        dataset = AudioDataset(mel_path=self.mel_path,
                               mel_ids=mel_ids,
                               text_dict=text_dict,
                               tokenizer=tokenizer)
        self.assertEqual(1, len(dataset))
        seq, mel, mel_id, mel_len = dataset[0]

        text = tokenizer.decode(seq)
        self.assertEqual('small sample text.', text)
        self.assertEqual((101, 80), mel.shape)
        self.assertEqual('small_sample', mel_id)
        self.assertEqual(101, mel_len)

    def test_collate_fn(self):
        mels = (np.zeros((2, 2), dtype=np.float),
                np.ones((3, 2), dtype=np.float))
        seqs = ([1, 2], [1, 2, 3])
        ids = ('mel_1', 'mel_2')
        mel_lens = (2, 3)
        batch = tuple(zip(seqs, mels, ids, mel_lens))

        seqs, mels, stops, ids, mel_lens = collate_fn(batch=batch, r=3)

        expected_seqs = np.array([[1, 2, 0], [1, 2, 3]])
        np.testing.assert_almost_equal(seqs, expected_seqs, decimal=8)

        expected_mels = np.array([[[-1, -1], [-1, -1], [0, 0]],
                                  [[1, 1], [1, 1], [1, 1]]])
        np.testing.assert_almost_equal(mels, expected_mels, decimal=8)

        expected_stops = np.array([[0, 1, 0], [0, 0, 1]])
        np.testing.assert_almost_equal(stops, expected_stops, decimal=8)

        self.assertEqual((2, 3), mel_lens)

