import abc
import json
import os
from copy import copy
from pathlib import Path

import numpy as np
from vlgp.constant import DEFAULT_VALUES


class Model:
    def __init__(self, config: dict):
        self.result_dir = Path(config.get('result_dir', os.path.curdir))

        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True)

        config_file = self.result_dir.joinpath('config.json')

        # fill unset config fields with existing ones
        if config_file.exists():
            old_config = json.load(config_file)
            old_config.update(config)

        self.config = copy(config)
        self.check_config()

        with config_file.open('w') as f:
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

    @abc.abstractmethod
    def check_config(self):
        pass


class VLGP(Model):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = self._build(config)

    def check_config(self):
        for k, v in DEFAULT_VALUES:
            self.config.setdefault(k, v)

    @staticmethod
    def _build(config):
        model = {'y_dim': config['y_dim'],
                 'z_dim': config['z_dim'],
                 'x_dim': config['x_dim'],
                 'likelihood': config['likelihood']}
        return model

    def infer(self, data):
        pass

    def vem(self, data):
        pass

    def save(self):
        np.save(self.result_dir.joinpath('model.npy'), self.model)
