from pathlib import Path
from typing import Union
from io import StringIO
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
    def from_string(cls, string: str) -> 'Config':

        with StringIO(string) as string_io:
            cfg = ruamel.yaml.load(string_io, Loader=ruamel.yaml.Loader)
            return Config(**cfg)

    def to_string(self) -> str:
        yaml = ruamel.yaml.YAML()
        with StringIO() as string_io:
            yaml.dump(self.__dict__, string_io)
            return string_io.getvalue()

    def update(self, new_cfg: 'Config') -> 'Config':
        """ Overrides training params """
        for p in self.training_params:
            if p in new_cfg.__dict__:
                self.__dict__[p] = getattr(new_cfg, p)
        return self


if __name__ == '__main__':
    cfg = Config(foo='bar')
    string = cfg.to_string()
    cfg_new = Config.from_string(string)
    print(cfg_new.__dict__)
