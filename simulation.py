import numpy as np
from numpy import pi, exp, sqrt, log2
import scipy.stats as stats


def sqexp(t, b):
    """Return autocorrelation.
    k(t, b) = exp(-b * t^2)

    Args:
        t : float
            lag.
        b : float
            scale.

    Returns:
        float
            correlation.
    """
    return exp(- b * t * t)


def spectral(w, b):
    """
    Spectral density of squared exponential covariance function
    :param w: float, frequency
    :param b: float, lengthscale
    :return: power
    """

    # TODO(yuan): change b to l^2

    return 0.5 * exp(-0.25 * w * w / b) / sqrt(pi * b)
    # return 0.5 * sqrt(2) * l * exp(-0.5 * w * w * l * l) / sqrt(pi)


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
    pN, _ = b.shape
    p = pN // N

    y = np.empty((T, N), dtype=float)
    Y = np.zeros((T, pN), dtype=float)
    if y0 is not None:
        for t in range(p):
            Y[t, :(p-t)*N] = y0[t:, :].flatten()

    for t in range(T):
        rate = np.exp(np.dot(Y[t, :], b) + np.dot(x[t, :], a))
        # y[:, t] = np.random.poisson(lambda_t)
        # truncate y to 1 if y > 1
        # it's equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        y[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate))
        if t + 1 < T:
            Y[t + 1, :] = np.roll(Y[t, :], -N)
            Y[t + 1, (p - 1) * N:] = y[t, :]

    return y, Y


def spikes2(x, a, b, c, y0=None, seed=None):
    """
    Simulate spike trains driven by latent processes
    :param x: (T, L), latent processes
    :param a: (L, N), coefficients of latent
    :param b: (p*N, N), coefficients of p-step history
    :param y0: (p, N), prehistory
    :param seed: random number seed
    :return: (T, N), spike trains
    """

    if seed is not None:
        np.random.seed(seed)

    T, L = x.shape
    _, N = a.shape
    pN, _ = b.shape
    p = pN // N

    y = np.zeros((T, N), dtype=float)
    rate = y.copy()
    Y = np.zeros((T, p*N), dtype=float)
    if y0 is not None:
        for t in range(p):
            Y[t, :(p-t)*N] = y0[t:, :].flatten()

    for t in range(T):
        rate[t, :] = np.exp(np.dot(Y[t, :], b) + np.dot(x[t, :], a) + c)
        # y[:, t] = np.random.poisson(lambda_t)
        # truncate y to 1 if y > 1
        # it's equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        # y[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate[t, :]))
        y[t, :] = 1 * (stats.poisson.rvs(rate[t, :]) > 0)
        if t + 1 < T:
            Y[t + 1, :] = np.roll(Y[t, :], -N)
            Y[t + 1, (p - 1) * N:] = y[t, :]

    return y, Y, rate

