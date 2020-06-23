import librosa
import numpy as np
import os
import unittest
from pathlib import Path
from preprocessing.audio import Audio
from utils.config import Config


class TestAudio(unittest.TestCase):

    def setUp(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = Path(current_dir)
        config_path = current_dir/'resources'/'test_config.yaml'
        self.wav_path = current_dir/'resources'/'small_sample.wav'
        self.cfg = Config.load(config_path)
        self.audio = Audio(self.cfg)

    def test_wav_to_mel(self):
        wav, sr = librosa.load(self.wav_path)
        mel = self.audio.wav_to_mel(wav)
        sum_mel = float(np.sum(mel))
        self.assertAlmostEqual(1691.0345458984375, sum_mel, places=6)
