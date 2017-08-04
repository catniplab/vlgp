import abc
import json
import os
from copy import copy
from pathlib import Path


class Model:
    def __init__(self, config):
        self.config = copy(config)
        self.result_dir = Path(config.get('result_dir', os.path.curdir))

        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=False)

        with self.result_dir.joinpath('config.json').open('w') as f:
            json.dump(self.config, f)

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def infer(self, data):
        pass

    @abc.abstractmethod
    def vem(self, data):
        pass


class VLGP(Model):
    def __init__(self):
        pass

    def infer(self, data):
        pass

    def vem(self, data):
        pass

    def save(self):
        pass
