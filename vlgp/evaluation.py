from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import numpy as np
import time
from numpy import empty_like
from scipy.stats import spearmanr

from .math import subspace
from .util import rotate, add_constant


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, fit):
        """
        Measure current fit

        Parameters
        ----------
        fit : dict
            current fit

        Returns
        -------
        list
            list of tuple (label, evaluation)
        """
        pass


class DefaultEvaluator(Evaluator):
    def __init__(self, x=None, a=None, b=None):
        self._x = x
        self._a = a
        self._b = b

    def evaluate(self, fit):
        a_true = self._a
        x_true = self._x
        loading_angle = latent_angle = spearman = None

        if a_true is not None:
            loading_angle = subspace(a_true.T, fit.a.T)
        if x_true is not None:
            ntrial, _, dyn_ndim = x_true.shape
            rotated = empty_like(x_true, dtype=float)
            for trial in range(ntrial):
                rotated[trial, ...] = rotate(add_constant(fit['mu'][trial, :]), x_true[trial, :])
            latent_angle = subspace(rotated.reshape((-1, dyn_ndim)), x_true.reshape((-1, dyn_ndim)))

            rho, _ = spearmanr(rotated.reshape((-1, dyn_ndim)), x_true.reshape((-1, dyn_ndim)))
            spearman = rho[np.arange(dyn_ndim), np.arange(dyn_ndim) + dyn_ndim]
        evaluation = [('latent_angle', latent_angle), ('loading_angle', loading_angle), ('spearman', spearman)]
        return evaluation


@contextmanager
def timer():
    tick = time.perf_counter()
    yield lambda : tock - tick
    tock = time.perf_counter()
