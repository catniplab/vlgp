from abc import ABCMeta, abstractmethod

from numpy import reshape, empty
from scipy.linalg import svd, lstsq
from sklearn.decomposition import FactorAnalysis


class Initializer(metaclass=ABCMeta):
    pass


class FAInitializer(Initializer):
    def __init__(self, dyn_ndim):
        self._dyn_ndim = dyn_ndim

    def init(self, y, h, mu0, a0, b0):
        dyn_ndim = self._dyn_ndim
        fa = FactorAnalysis(n_components=dyn_ndim, svd_method='lapack')
        ntrial, nbin, obs_ndim = y.shape

        if a0 is None:
            if  mu0 is None:
                mu = fa.fit_transform(y.reshape((-1, obs_ndim)))
                a = fa.components_
            else:
                mu = mu0
                a = lstsq(mu.reshape((-1, obs_ndim)), y.reshape((-1, obs_ndim)))[0]
        else:
            a = a0
            if mu0 is None:
                    mu = lstsq(a.T, y.reshape((-1, obs_ndim)).T)[0].T.reshape((ntrial, nbin, dyn_ndim))
            else:
                mu = mu0

        U, s, Vh = svd(a, full_matrices=False)
        mu = reshape(mu @ a @ Vh.T, (ntrial, nbin, dyn_ndim))
        a = Vh

        if b0 is None:
            lag1 = h.shape[-1]
            b = empty((lag1, obs_ndim), dtype=float)
            for obs_dim in range(obs_ndim):
                b[:, obs_dim] = \
                    lstsq(h.reshape((obs_ndim, -1, lag1))[obs_dim, :], y.reshape((-1, obs_ndim))[:, obs_dim])[0]
        else:
            b = b0

        return mu, a, b
