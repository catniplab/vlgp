import numpy as np
from scipy.linalg import toeplitz

from vlgp.math import ichol_gauss, orth, rectify


def test_ichol_gauss():
    n = 500
    omega = 1
    dsq = np.arange(n) ** 2
    k = np.exp(- omega * dsq)
    K = toeplitz(k)
    G = ichol_gauss(n, omega, n)
    assert np.allclose(K, G @ G.T)


def test_orth():
    n = 500
    p = 200
    q = 100
    x = np.random.randn(n, q)
    a = np.random.rand(q, p)
    x_orth, a_orth = orth(x, a)
    assert np.allclose(x @ a, x_orth @ a_orth)


def test_rectify():
    n = 5000
    x = np.random.randn(n)
    assert np.array_equal(rectify(x), np.maximum(0, x))
