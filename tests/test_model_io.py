import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim.adam import Adam

from model.io import save_model, load_model
from model.tacotron import Tacotron
from utils.config import Config


class TestModelIO(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory(prefix='TestModelIO')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = Path(current_dir)
        self.config_path = current_dir/'resources'/'test_config.yaml'

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_save_load(self) -> None:
        tmp_dir = Path(self.tmp_dir.name)
        cfg = Config.load(self.config_path)
        model = Tacotron.from_config(cfg)
        opti = Adam(model.parameters(), lr=2e-5)
        save_model(model, opti, cfg, tmp_dir/'model.zip')

        model_2, opti_2, cfg_2 = load_model(tmp_dir/'model.zip')
        self._assert_equal_models(model, model_2)
        for param_group in opti_2.param_groups:
            self.assertAlmostEqual(2e-5, param_group['lr'], places=10)
        self.assertEqual('english_cleaners', cfg_2.cleaners)

    @staticmethod
    def _assert_equal_models(model_1, model_2):
        items_1 = model_1.state_dict().items()
        items_2 = model_2.state_dict().items()
        for key_item_1, key_item_2 in zip(items_1, items_2):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                raise ValueError
