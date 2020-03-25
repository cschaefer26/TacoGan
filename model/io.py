import shutil
from pathlib import Path

import torch
from torch.optim.adam import Adam

from model.gan import GAN
from model.tacotron_new import Tacotron
from utils.config import Config
from utils.io import create_dir


class ModelPackage:
    """
    Container class for models and optimizers to ease IO
    """

    def __init__(self,
                 tacotron: Tacotron,
                 gan: GAN,
                 taco_opti: Adam,
                 gen_opti: Adam,
                 disc_opti: Adam,
                 cfg: Config) -> None:
        self.tacotron = tacotron
        self.gan = gan
        self.taco_opti = taco_opti
        self.gen_opti = gen_opti
        self.disc_opti = disc_opti
        self.cfg = cfg

    def save(self, path: Path) -> None:
        path = path.parent / path.stem
        tmp_dir = Path(str(path) + '_save_tmp')
        create_dir(tmp_dir, overwrite=True)
        torch.save(self.tacotron.state_dict(), tmp_dir/'tacotron.pyt')
        torch.save(self.gan.state_dict(), tmp_dir/'gan.pyt')
        torch.save(self.taco_opti.state_dict(), tmp_dir/'taco_opti.pyt')
        torch.save(self.gen_opti.state_dict(), tmp_dir/'gen_opti.pyt')
        torch.save(self.disc_opti.state_dict(), tmp_dir/'disc_opti.pyt')
        self.cfg.save(tmp_dir/'config.yaml')
        shutil.make_archive(path, 'zip', tmp_dir)
        shutil.rmtree(tmp_dir)

    @classmethod
    def load(cls, path: Path, device='cpu'):
        device = torch.device(device)
        tmp_dir = Path(str(path) + '_load_tmp')
        shutil.unpack_archive(str(path), extract_dir=tmp_dir)
        cfg = Config.load(tmp_dir / 'config.yaml')

        tacotron = Tacotron.from_config(cfg).to(device)
        state_dict = torch.load(tmp_dir / 'tacotron.pyt', device)
        tacotron.load_state_dict(state_dict, strict=False)

        gan = GAN.from_config(cfg).to(device)
        state_dict = torch.load(tmp_dir / 'gan.pyt', device)
        gan.load_state_dict(state_dict, strict=False)

        taco_opti = Adam(tacotron.parameters())
        state_dict = torch.load(tmp_dir / 'taco_opti.pyt', device)
        taco_opti.load_state_dict(state_dict)

        gen_opti = Adam(gan.generator.parameters())
        state_dict = torch.load(tmp_dir / 'gen_opti.pyt', device)
        gen_opti.load_state_dict(state_dict)

        disc_opti = Adam(gan.discriminator.parameters())
        state_dict = torch.load(tmp_dir / 'disc_opti.pyt', device)
        disc_opti.load_state_dict(state_dict)

        model_package = ModelPackage(
            tacotron=tacotron, gan=gan, taco_opti=taco_opti,
            gen_opti=gen_opti, disc_opti=disc_opti, cfg=cfg)
        shutil.rmtree(tmp_dir)

        return model_package

