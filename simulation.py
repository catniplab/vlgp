import numpy as np

from numpy import pi, exp, sqrt, log2, ceil, fft
from numpy import zeros, arange, empty, ones, roll, empty_like
from numpy.random import random, multivariate_normal
from scipy import stats


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


def latents(L, T, std, b, dt=1.0, seed=None):
    """Simulate Gaussian processes

    Args:
        L: number of processes
        T: duration
        std: standard deviation
        b: scale
        dt: unit
        seed: random number seed

    Returns:
        x: simulation (T, L)
        ticks: ticks (N,)
    """

    if seed is not None:
        np.random.seed(seed)

    x = zeros((T, L), dtype=float)

    M = int(2 ** ceil(log2(T)))
    T0 = M * dt
    dw = 2 * pi / T0
    wu = 2 * pi / dt
    t = arange(0, T0, dt)
    w = arange(0, wu, dw)

    for l in range(L):
        B = 2 * sqrt(spectral(w, b) * dw) * exp(1j * random(M) * 2 * pi)
        B[0] = 0
        x[:, l] = std * T * fft.ifft(B, T).real

    return x, t[:T]


def spikes(latent, alpha, beta, y0=None, seed=None):
    """Simulate spike trains driven by latent processes

    Args:
        latent: latent processes (T, L)
        alpha: coefficients of latent (L, N)
        beta: coefficients of regression (1 + p*N, N)
        y0: observations before epoch
        seed:

    Returns:
        spike trains (T, N)
    """

    if seed is not None:
        np.random.seed(seed)

    T, L = latent.shape
    _, N = alpha.shape
    k, _ = beta.shape
    p = (beta.shape[0] - 1) // N

    spike = empty((T, N), dtype=float)
    regressor = ones((T, k), dtype=float)
    rate = spike.copy()
    if y0 is not None:
        for t in range(p):
            regressor[t, 1:(p - t) * N] = y0[t:, :].flatten()

    for t in range(T):
        eta = regressor[t, :].dot(beta) + latent[t, :].dot(alpha)
        rate[t, :] = exp(eta.clip(-30, 30))
        # spike[t, :] = (np.random.poisson(rate[t, :], size=(1, N)) > 0) * 1
        # truncate spike to 1 if spike > 1
        # it's equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        spike[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate[t, :]))
        # spike[t, :] = 1 * (stats.poisson.rvs(rate[t, :]) > 0)
        if t + 1 < T and p != 0:
            regressor[t + 1, 1:] = roll(regressor[t, 1:], -N)
            regressor[t + 1, 1 + (p - 1) * N:] = spike[t, :]

    return spike, regressor, rate


def gaussian(latent, a, b, K, y0=None, seed=None):
    """Simulate Gaussian observations driven by latent processes

    Args:
        latent: latent processes (T, L)
        a: coefficients of latent (L, N)
        b: coefficients of regression (1 + p*N, N)
        K:
        y0:
        seed:

    Returns:

    """
    if seed is not None:
        np.random.seed(seed)

    T, L = latent.shape
    _, N = a.shape
    p = (b.shape[0] - 1) // N

    y = empty((T, N), dtype=float)
    mu = empty_like(y, dtype=float)
    h = ones((T, b.shape[0]), dtype=float)
    if y0 is not None:
        for t in range(p):
            h[t, 1:(p - t) * N] = y0[t:, :].flatten()

    for t in range(T):
        mu[t, :] = latent[t, :].dot(a) + h[t, :].dot(b)
        y[t, :] = multivariate_normal(mu[t, :], K)
        if t + 1 < T and p != 0:
            h[t + 1, 1:] = roll(h[t, 1:], -N)
            h[t + 1, 1 + (p - 1) * N:] = y[t, :]

    return y, h, mu
