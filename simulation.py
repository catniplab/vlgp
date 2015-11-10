import numpy as np

from numpy import pi, exp, sqrt, log2, ceil, fft, einsum
from numpy import zeros, arange, empty, ones, roll, empty_like
from numpy.random import random, multivariate_normal
from scipy import stats

# lower and upper bound of exp
LB = -20
UB = 20


def sqexp(t, w):
    """Squared exponential correlation

    Args:
        t: lag
        w: inverse of squared lengthscale

    Returns:

    """

    return exp(- w * t ** 2)


def spectral(f, w):
    """Spectral density of squared exponential covariance function

    Args:
        f: frequency
        w: inverse of squared lengthscale

    Returns:

    """

    # TODO(yuan): change w to l^2

    return 0.5 * exp(- 0.25 * f * f / w) / sqrt(pi * w)


def gp(b, T, std, dt=1.0, seed=None):
    """Simulate Gaussian processes

    Args:
        b: scale
        T: duration
        std: standard deviation
        dt: unit
        seed: random number seed

    Returns:
        x: simulation (T, L)
        ticks: ticks (N,)
    """

    if seed is not None:
        np.random.seed(seed)

    x = zeros((T, b.shape[0]), dtype=float)

    M = int(2 ** ceil(log2(T)))
    T0 = M * dt
    dw = 2 * pi / T0
    wu = 2 * pi / dt
    t = arange(0, T0, dt)
    w = arange(0, wu, dw)

    for l in range(b.shape[0]):
        B = 2 * sqrt(spectral(w, b[l]) * dw) * exp(1j * random(M) * 2 * pi)
        B[0] = 0
        x[:, l] = std * T * fft.ifft(B, T).real

    return x, t[:T]


def spikes(x, a, b, y0=None, seed=None):
    """Simulate y trains driven by x processes

    Args:
        x: x processes (T, L)
        a: coefficients of x (L, N)
        b: coefficients of regression (1 + p*N, N)
        y0: observations before epoch (p, N)
        seed: random seed

    Returns:
        y: spike trains (T, N)
        h: regressor (T, 1 + p*N)
        rate: firing rates (T, N)
    """

    if seed is not None:
        np.random.seed(seed)

    T, L = x.shape
    _, N = a.shape
    p = (b.shape[0] - 1) // N

    y = empty((T, N), dtype=float)
    h = ones((N, T, b.shape[0]), dtype=float)
    rate = empty_like(y, dtype=float)
    if y0 is not None:
        for t in range(p):
            h[:, t, 1:p - t + 1] = y0[t:, :].T

    for t in range(T):
        eta = einsum('ij, ji -> i', h[:, t, :], b) + x[t, :].dot(a)
        rate[t, :] = exp(eta.clip(LB, UB))
        # truncate y to 1 if y > 1
        # equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        y[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate[t, :]))
        # y[t, :] = 1 * (stats.poisson.rvs(rate[t, :]) > 0)
        if t + 1 < T and p != 0:
            h[:, t + 1, 1:] = roll(h[:, t, 1:], -1, axis=1)
            h[:, t + 1, -1] = y[t, :]

    return y, h, rate


def lfp(x, a, b, K, y0=None, seed=None):
    """Simulate Gaussian observations driven by x processes

    Args:
        x: x processes (T, L)
        a: coefficients of x (L, N)
        b: coefficients of regression (1 + p*N, N)
        K: noise
        y0: observations before epoch
        seed: random seed

    Returns:
        y: LFP
        h: regressor
        mu: mean
    """
    if seed is not None:
        np.random.seed(seed)

    T, L = x.shape
    _, N = a.shape
    p = (b.shape[0] - 1) // N

    y = empty((T, N), dtype=float)
    mu = empty_like(y, dtype=float)
    h = ones((N, T, b.shape[0]), dtype=float)
    if y0 is not None:
        for t in range(p):
            h[:, t, 1:p - t + 1] = y0[t:, :].T

    for t in range(T):
        mu[t, :] = x[t, :].dot(a) + einsum('ij, ji -> i', h[:, t, :], b)
        y[t, :] = multivariate_normal(mu[t, :], K)
        if t + 1 < T and p != 0:
            h[:, t + 1, 1:] = roll(h[:, t, 1:], -1, axis=1)
            h[:, t + 1, -1] = y[t, :]

    return y, h, mu
