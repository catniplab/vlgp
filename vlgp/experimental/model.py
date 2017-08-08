# A job contains current values of parameters, hyperparameters and latents.
# Latents are trial-wise, and (hyper-)parameters are model-wise.

import abc
import json
import os
import gc
from copy import copy
from pathlib import Path

import numpy as np
from scipy.linalg import norm

from vlgp.constant import DEFAULT_VALUES
from vlgp.evaluation import timer
from vlgp.experimental.core import estep, mstep, hstep


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
        # callbacks = []
        tol = self.config['tol']
        niter = self.config['niter']

        job = {'model': self.model, 'trials': data}
        # initialize a job

        self.initialize(job)

        #######################
        # iterative algorithm #
        #######################

        # disable gabbage collection during the iterative procedure
        gc.disable()
        for it in range(job['it'], niter):
            job['it'] += 1

            ##########
            # E step #
            ##########
            with timer() as estep_elapsed:
                estep(job)

            ##########
            # M step #
            ##########
            with timer() as mstep_elapsed:
                mstep(job)

            ###################
            # hyperparam step #
            ###################
            with timer() as hstep_elapsed:
                hstep(job)

            job['e_elapsed'].append(estep_elapsed())
            job['m_elapsed'].append(mstep_elapsed())
            job['h_elapsed'].append(hstep_elapsed())

            # for callback in callbacks:
            #     try:
            #         callback(model)
            #     finally:
            #         pass

            #####################
            # convergence check #
            #####################
            # mu = model['mu']
            a = self.model['a']
            b = self.model['b']
            # dmu = model['dmu']
            da = job['da']
            db = job['db']

            converged = norm(da) < tol * norm(a) and norm(db) < tol * norm(b)
            should_stop = converged

            if should_stop:
                break

        gc.enable()  # enable gabbage collection

        ##############################
        # end of iterative procedure #
        ##############################

    def initialize(self, job):
        job.setdefault('it', 0)
        job.setdefault('e_elapsed', [])
        job.setdefault('m_elapsed', [])
        job.setdefault('h_elapsed', [])
        job.setdefault('em_elapsed', [])
        job.setdefault('dir', self.result_dir)

        trials = job['trials']
        # TODO: build arrays of latents, parameters and hyperparameters

    def save(self):
        np.save(self.result_dir.joinpath('model.npy'), self.model)
