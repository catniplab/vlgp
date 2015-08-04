from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, log2, dot
import scipy.stats as stats


def spectral(w, b):
    """
    Spectral density of squared exponential covariance function
    :param w: float, frequency
    :param b: float, lengthscale
    :return: power
    """

    # TODO(yuan): change b to l^2

    return 0.5 * exp(-0.25 * w * w / b) / sqrt(pi * b)


def gpspectral(sigma, b, n, dt=1.0, seed=None):
    """
    Simulate univariate Gaussian process
    :param sigma: float, standard deviation
    :param b: float, lengthscale
    :param n: int, number of data points
    :param dt: float, unit time interval
    :param seed: optional, random number seed
    :return: (n,), process
    """

    if seed is not None:
        np.random.seed(seed)

    m = int(2 ** np.ceil(log2(n)))  # needs to be power of 2
    t0 = m * dt
    dw = 2 * pi / t0
    wu = 2 * pi / dt
    t = np.arange(0, t0, dt)
    w = np.arange(0, wu, dw)

    B = 2 * sqrt(spectral(w, b) * dw) * exp(1j * np.random.rand(m) * 2 * pi)
    B[0] = 0
    return sigma * n * np.fft.ifft(B, n).real, t[:n]


def latents(L, T, std, b, dt=1.0, seed=None):
    """
    Simulate multiple Gaussian processes
    :param L: int, number of processes
    :param T: int, time length
    :param std: float, standard deviation
    :param b: float, scale
    :param dt: float, optional, unit time interval
    :param seed: int, optional, random number seed
    :return
        x: array(T, L), processes
        ticks: array(N,), ticks
    """

    if seed is not None:
        np.random.seed(seed)

    x = np.zeros((T, L), dtype=float)

    M = int(2 ** np.ceil(log2(T)))
    T0 = M * dt
    dw = 2 * pi / T0
    wu = 2 * pi / dt
    t = np.arange(0, T0, dt)
    w = np.arange(0, wu, dw)

    for l in range(L):
        B = 2 * sqrt(spectral(w, b) * dw) * exp(1j * np.random.rand(M) * 2 * pi)
        B[0] = 0
        x[:, l] = std * T * np.fft.ifft(B, T).real

    return x, t[:T]


def multi_spike(gamma, x, A, B, y0=None, seed=None):
    """Simulate multiple spike processes driven by latent processes.
    lam_t = exp(mu + Ax_t + sum_i^p B_i y_{t-i})

    Args:
        mu : array
            (n, 1) background mean.
        x : array
            (m, N) m-variate latent GP process, length of N, increasing order of t.
        A : array
            (n, m) mapping x_t.
        B : array
            (n, n, p) mapping y_{t-1}, ..., y_{t-p}, increasing order of t, [t-p, ..., t-1].
        y0 : array, optional
            (n, p) starting value before time 1. 0 (default).
        seed : int, optional
            random number seed.

    Returns:
        array
            (n, N) simulation.
    """

    if seed is not None:
        np.random.seed(seed)
    T, L = x.shape
    _, N, p = B.shape
    y = np.empty((T, N), dtype=int)
    if y0 is None:
        y0 = np.zeros((p, N), dtype=int)

    for t in range(T):
        Ax = dot(A, x[t, :])
        sum_By = 0.0
        for i in range(p):
            if t - p + i >= 0:
                sum_By += dot(B[:, :, i], y[t - p + i, :])
            else:
                sum_By += dot(B[:, :, i], y0[t + i, :])
        lam_t = exp(gamma + Ax + sum_By)
        # y[:, t] = np.random.poisson(lambda_t)
        # truncate y to 1 if y > 1
        # it's equivalent to Bernoulli P(=1) = (1 - e^-(lam_t))
        y[t, :] = stats.bernoulli.rvs(1.0 - exp(- lam_t))
    return y


def spikes(x, a, b, y0=None, seed=None):
    """
    Simulate spike trains driven by latent processes
    :param x: (T, L), latent processes
    :param a: (L, N), coefficients of latent
    :param b: (1 + p*N, N), coefficients of p-step history
    :param y0: (p, N), prehistory
    :param seed: random number seed
    :return: (T, N), spike trains
    """

    if seed is not None:
        np.random.seed(seed)

    T, L = x.shape
    _, N = a.shape
    rb, _ = b.shape
    p = (rb - 1)/N

    y = np.empty((T, N), dtype=float)
    Y = np.zeros((T, 1 + p*N), dtype=float)
    Y[:, 0] = 1
    if y0 is not None:
        for t in range(p):
            Y[t, 1:(p-t)*N] = y0[t:, :].flatten()

    for t in range(T):
        rate = np.exp(np.dot(Y[t, :], b) + np.dot(x[t, :], a))
        # y[:, t] = np.random.poisson(lambda_t)
        # truncate y to 1 if y > 1
        # it's equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        y[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate))
        if t + 1 < T:
            Y[t + 1, 1:] = np.roll(Y[t, 1:], -N)
            Y[t + 1, 1 + (p - 1) * N:] = y[t, :]

    return y, Y
