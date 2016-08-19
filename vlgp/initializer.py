from abc import ABCMeta, abstractmethod

from numpy import reshape
from scipy.linalg import svd
from sklearn.decomposition import FactorAnalysis


class Initializer(metaclass=ABCMeta):
    pass


class FAInitializer(Initializer):
    def __init__(self, dyn_dim):
        self._dyn_dim = dyn_dim

    def init(self, obs):
        dyn_dim = self._dyn_dim
        fa = FactorAnalysis(n_components=dyn_dim, svd_method='lapack')
        ntrial, nbin, obs_dim = obs.shape[-1]
        mu = fa.fit_transform(obs.reshape((-1, obs_dim)))
        a = fa.components_
        U, s, Vh = svd(a, full_matrices=False)
        mu = reshape(mu @ a @ Vh.T, (ntrial, nbin, dyn_dim))
        a = Vh
        return mu, a
