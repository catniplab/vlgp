from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import empty_like
from scipy.stats import spearmanr

from vlgp import subspace, rotate, add_constant


class Profiler(metaclass=ABCMeta):
    @abstractmethod
    def profile(self, fit):
        pass


class DefaultProfiler(Profiler):
    def __init__(self, x=None, a=None, b=None):
        self._x = x
        self._a = a
        self._b = b

    def profile(self, fit):
        statistics = {}
        a_true = self._a
        x_true = self._x

        if a_true is not None:
            subspace_loading = subspace(a_true.T, fit.a.T)
            statistics['subspace_loading'] = subspace_loading
        if x_true is not None:
            ntrial, _, dyn_ndim = x_true.shape
            rotated = empty_like(x_true, dtype=float)
            for trial in range(ntrial):
                rotated[trial, ...] = rotate(add_constant(fit['mu'][trial, :]), x_true[trial, :])
            subspace_dyn = subspace(rotated.reshape((-1, dyn_ndim)), x_true.reshape((-1, dyn_ndim)))
            statistics['subspace_dyn'] = subspace_dyn

            rho, _ = spearmanr(rotated.reshape((-1, dyn_ndim)), x_true.reshape((-1, dyn_ndim)))
            rankcorr_dyn = rho[np.arange(dyn_ndim), np.arange(dyn_ndim) + dyn_ndim]
            statistics['rankcorr_dyn'] = rankcorr_dyn

        return statistics
