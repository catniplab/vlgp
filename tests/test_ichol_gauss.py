from unittest import TestCase

import numpy as np
from scipy.linalg import toeplitz

from vlgp.math import ichol_gauss2


class TestIcholGauss(TestCase):
    def test_ichol_gauss(self):
        np.random.seed(0)
        n = 500
        omega = 1
        dsq = np.arange(n) ** 2
        k = np.exp(- omega * dsq)
        K = toeplitz(k)
        G = ichol_gauss2(n, omega)
        self.assertTrue(np.allclose(K, G @ G.T))
