import abc
from abc import ABCMeta, abstractmethod
import numpy as np


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def loading(self):
        pass

    @abstractmethod
    def posterior(self):
        pass

    @abstractmethod
    def regression(self):
        pass


class NewtonOptimizer(Optimizer):
    def __init__(self, model, adjust=True, eps=1e-8):
        self._adjust = adjust
        self._ss_posterior_gradient = np.full_like(model['mu'].shape,
                                                   fill_value=eps)  # gradient sum of squares, posterior mean
        self._ss_loading_gradient = np.full_like(model['a'].shape,
                                                 fill_value=eps)  # gradient sum of squares, loading matrix
        self._ss_regression_gradient = np.full_like(model['b'].shape,
                                                    fill_value=eps)  # gradient sum of squares, regression

    def loading(self):
        pass

    def posterior(self):
        pass

    def regression(self):
        pass


class GradientDescentOptimizer(Optimizer):
    def __init__(self):
        pass

    def loading(self):
        pass

    def posterior(self):
        pass

    def regression(self):
        pass
