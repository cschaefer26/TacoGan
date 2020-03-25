import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim.adam import Adam

from model.gan import GAN
from model.io import ModelPackage
from model.tacotron_new import Tacotron
from utils.config import Config


class TestModelPackage(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory(prefix='TestModelPackage')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = Path(current_dir)
        self.config_path = current_dir/'resources'/'test_config.yaml'

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_save_load(self) -> None:
        tmp_dir = Path(self.tmp_dir.name)
        cfg = Config.load(self.config_path)
        tacotron = Tacotron.from_config(cfg)
        taco_opti = Adam(tacotron.parameters(), lr=2e-5)
        gan = GAN.from_config(cfg)
        gen_opti = Adam(gan.generator.parameters(), lr=3e-5)
        disc_opti = Adam(gan.discriminator.parameters())
        model_package = ModelPackage(
            tacotron=tacotron, gan=gan, taco_opti=taco_opti,
            gen_opti=gen_opti, disc_opti=disc_opti, cfg=cfg)
        model_package.save(tmp_dir/'model.zip')

        m = ModelPackage.load(tmp_dir/'model.zip')
        self._assert_equal_models(m.tacotron, tacotron)
        self._assert_equal_models(m.gan, gan)
        for param_group in m.taco_opti.param_groups:
            self.assertAlmostEqual(2e-5, param_group['lr'], places=10)
        self.assertEqual('english_cleaners', m.cfg.cleaners)

    @staticmethod
    def _assert_equal_models(model_1, model_2):
        items_1 = model_1.state_dict().items()
        items_2 = model_2.state_dict().items()
        for key_item_1, key_item_2 in zip(items_1, items_2):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                raise ValueError
