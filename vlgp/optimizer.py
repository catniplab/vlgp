from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, grad):
        pass


class AdamOptimizer(Optimizer):
    def __init__(self, ndim, learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8):
        """

        Parameters
        ----------
        ndim : int
            length of updating vector
        learning_rate : double
            it is what it is
        b1 : double
            decay rate of the first moment
        b2 : double
            decay rate of the second moment
        eps : double
            a small positive number
        """
        self._learning_rate = learning_rate
        self._b1 = b1
        self._b2 = b2
        self._eps = eps
        self._counter = 0  # count how many updates have been computed
        self._m = [np.zeros(ndim)]  # first moment
        self._v = [np.zeros(ndim)]  # second moment

    def update(self, grad):
        self._counter += 1
        current_m = self._b1 * self._m[self._counter - 1] + (1 - self._b1) * grad
        current_v = self._b2 * self._v[self._counter - 1] + (1 - self._b2) * grad ** 2
        self._m.append(current_m)
        self._v.append(current_v)
        m_hat = current_m / (1 - self._b1 ** self._counter)
        v_hat = current_v / (1 - self._b2 ** self._counter)
        return self._learning_rate * m_hat / (np.sqrt(v_hat) + self._eps)


class AdagradOptimizer(Optimizer):
    def __init__(self, ndim, learning_rate=0.001, b=0.999, eps=1e-8):
        """

        Parameters
        ----------
        ndim : int
            length of updating vector
        learning_rate : double
            it is what it is
        b : double
            decay rate of the second moment
        eps : double
            a small positive number
        """
        self._learning_rate = learning_rate
        self._b = b
        self._eps = eps
        self._counter = 0
        self._v = [np.zeros(ndim)]  # second moment

    def update(self, grad):
        self._counter += 1
        current_v = self._b * self._v[self._counter - 1] + (1 - self._b) * grad ** 2
        self._v.append(current_v)
        return self._learning_rate * grad / (np.sqrt(current_v) + self._eps)
