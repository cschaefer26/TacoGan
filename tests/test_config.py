import unittest
from pathlib import Path
from utils.config import Config
import tempfile


class TestConfig(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory(prefix='TestConfigIO_config.yaml')

    def test_save_load(self):
        tmp_dir = Path(self.tmp_dir.name)
        cfg = Config(sample_rate=22050)
        cfg.save(tmp_dir/'config.yaml')
        cfg = Config.load(tmp_dir/'config.yaml')
        self.assertEqual(22050, cfg.sample_rate)

    # noinspection PyUnresolvedReferences
    def test_override(self):
        cfg = Config(sample_rate=22050, steps_to_eval=10)
        cfg_2 = Config(sample_rate=40000, steps_to_eval=100)
        cfg.update(cfg_2)
        self.assertEqual(22050, cfg.sample_rate)
        self.assertEqual(100, cfg.steps_to_eval)





