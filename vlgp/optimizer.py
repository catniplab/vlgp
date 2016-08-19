from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, grad):
        pass

    @staticmethod
    def new(cls, **kwargs):
        return cls(**kwargs)


class AdamOptimizer(Optimizer):
    def __init__(self, dim, b1=0.9, b2=0.999, eps=1e-8):
        self._b1 = b1
        self._b2 = b2
        self._eps = eps
        self._counter = 0
        self._m = [np.zeros(dim)]  # first moment
        self._v = [np.zeros(dim)]  # second moment

    def update(self, grad):
        self._counter += 1
        current_m = self._b1 * self._m[self._counter - 1] + (1 - self._b1) * grad
        current_v = self._b2 * self._v[self._counter - 1] + (1 - self._b2) * grad ** 2
        self._m.append(current_m)
        self._v.append(current_v)
        m_hat = current_m / (1 - self._b1 ** self._counter)
        v_hat = current_v / (1 - self._b2 ** self._counter)
        return m_hat / (np.sqrt(v_hat) + self._eps)


class AdagradOptimizer(Optimizer):
    def __init__(self, dim, b=0.999, eps=1e-8):
        self._b = b
        self._eps = eps
        self._counter = 0
        self._v = [np.zeros(dim)]  # second moment

    def update(self, grad):
        self._counter += 1
        current_v = self._b * self._v[self._counter - 1] + (1 - self._b) * grad ** 2
        self._v.append(current_v)
        return grad / (np.sqrt(current_v) + self._eps)
