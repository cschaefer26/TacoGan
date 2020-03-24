from pathlib import Path
from typing import Union

import ruamel.yaml


class Config:

    def __init__(self, **kwargs) -> None:
        for name in kwargs:
            setattr(self, name, kwargs[name])

        # training params can be overriden
        self.training_params = [
            'training_schedule', 'log_dir',
            'steps_to_eval', 'steps_to_checkpoint']

    def __getattr__(self, item):
        """ Workaround to switch off access warnings for non-existing attributes """
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError(f'Config does not contain item: {item}!')

    @classmethod
    def load(cls, path: Union[Path, str]):
        with open(str(path), 'r', encoding='utf-8') as f:
            cfg = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)
            return Config(**cfg)

    def save(self, path: Path) -> None:
        yaml = ruamel.yaml.YAML()
        with open(str(path), 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f)

    def update(self, new_cfg: 'Config') -> 'Config':
        """ Overrides training params """
        for p in self.training_params:
            if p in new_cfg.__dict__:
                self.__dict__[p] = getattr(new_cfg, p)
        return self
