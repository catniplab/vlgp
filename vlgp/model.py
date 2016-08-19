from abc import ABCMeta, abstractmethod
from .optimizer import AdamOptimizer


class Model(metaclass=ABCMeta):
    pass


class ModelFit(metaclass=ABCMeta):
    pass


class VLGPModel(Model):
    def __init__(self,
                 dyn_dim,
                 obs_dim,
                 ntrial,
                 hyperparam):
        self._dyn_dim = dyn_dim
        self._obs_dim = obs_dim
        self._ntrial = ntrial
        self._hyperparam = hyperparam
        pass

    def fit(self,
            data,
            learning_rate=0.001,
            optimizer_class=AdamOptimizer,
            optimizer_args=None,
            hessian=True,
            regular_hessian=False,
            eps=1e-8,
            rtol=1e-5,
            atol=1e-8):
        dyn_dim = self._dyn_dim
        obs_dim = self._obs_dim
        ntrial = self._ntrial
        hyperparam = self._hyperparam


class VLGPModelFit(ModelFit):
    pass
