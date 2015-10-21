import numpy as np
from scipy import stats
from numpy import pi, exp, sqrt, log2


def sqexp(t, w):
    """
    Squared exponential correlation
    k(t, w) = exp(-w * t^2)
    :param t: lag
    :param w: inverse of squared lengthscale
    :return:
    """

    return exp(- w * t ** 2)


def spectral(f, w):
    """
    Spectral density of squared exponential covariance function
    :param f: frequency
    :param w: inverse of squared lengthscale
    :return: power
    """
    # TODO(yuan): change w to l^2

    return 0.5 * exp(-0.25 * f * f / w) / sqrt(pi * w)
    # return 0.5 * sqrt(2) * l * exp(-0.5 * f * f * l * l) / sqrt(pi)


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
        B = 2 * sqrt(spectral(w, b) * dw) * exp(1j * np.random.random(M) * 2 * pi)
        B[0] = 0
        x[:, l] = std * T * np.fft.ifft(B, T).real

    return x, t[:T]


def spikes(latent, alpha, beta, intercept=True, y0=None, seed=None):
    """
    Simulate spike trains driven by latent processes
    :param latent: (T, L), latent processes
    :param alpha: (L, N), coefficients of latent
    :param beta: (1 + p*N, N), coefficients of p-step makeregressor
    :param y0: (p, N), prehistory
    :param seed: random number seed
    :return: (T, N), spike trains
    """

    if seed is not None:
        np.random.seed(seed)

    T, L = latent.shape
    _, N = alpha.shape
    k, _ = beta.shape
    p = (k - intercept) // N

    spike = np.empty((T, N), dtype=float)
    regressor = np.ones((T, k), dtype=float)
    rate = spike.copy()
    if y0 is not None:
        for t in range(p):
            regressor[t, intercept:(p - t) * N] = y0[t:, :].flatten()

    for t in range(T):
        eta = np.clip(np.dot(regressor[t, :], beta) + np.dot(latent[t, :], alpha), -30, 30)
        rate[t, :] = np.exp(eta)
        # spike[t, :] = (np.random.poisson(rate[t, :], size=(1, N)) > 0) * 1
        # truncate spike to 1 if spike > 1
        # it's equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        spike[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate[t, :]))
        # spike[t, :] = 1 * (stats.poisson.rvs(rate[t, :]) > 0)
        if t + 1 < T and p != 0:
            regressor[t + 1, intercept:] = np.roll(regressor[t, intercept:], -N)
            regressor[t + 1, intercept + (p - 1) * N:] = spike[t, :]

    return spike, regressor, rate


def gaussian(latent, a, b, K, y0=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T, L = latent.shape
    _, N = a.shape
    p = (b.shape[0] - 1) // N

    observation = np.empty((T, N), dtype=float)
    mean = np.empty_like(observation, dtype=float)
    h = np.ones((T, b.shape[0]), dtype=float)
    if y0 is not None:
        for t in range(p):
            h[t, 1:(p - t) * N] = y0[t:, :].flatten()

    for t in range(T):
        mean[t, :] = latent[t, :].dot(a) + h[t, :].dot(b)
        observation[t, :] = np.random.multivariate_normal(mean[t, :], K)
        if t + 1 < T and p != 0:
            h[t + 1, 1:] = np.roll(h[t, 1:], -N)
            h[t + 1, 1 + (p - 1) * N:] = observation[t, :]

    return observation, h, mean
