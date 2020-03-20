import os
from pathlib import Path


class Paths:

    def __init__(self):

        self.ckpt = Path('checkpoints').expanduser().resolve()
        self.data = Path('data').expanduser().resolve()
        self.mel = self.data / 'mel'
        self.create_paths()

    def create_paths(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.ckpt, exist_ok=True)

