"""
Functions of simulation
Gaussian process
Spike train
"""
import numpy as np
from numpy.random import random, multivariate_normal
from scipy import stats
from scipy.linalg import toeplitz

from .math import sexp, identity, ichol_gauss


def sqexp(t, omega):
    """Squared exponential correlation

    Args:
        t: lag
        omega: inverse of squared lengthscale

    Returns:
        correlation
    """

    return np.exp(- omega * t ** 2)


# def spectral(f, omega):
#     """Spectral density of squared exponential covariance function
#
#     Args:
#         f: frequency
#         omega: inverse of squared lengthscale
#
#     Returns:
#         power
#     """
#
#     return 0.5 * np.exp(- 0.25 * f * f / omega) / np.sqrt(np.pi * omega)
#     # return np.exp(- f * f / omega)


def gp_ichol(omega, nbin, std, r=None, tol=1e-6, seed=None):
    """Simulate Gaussian processes by incomplete Choleksy factorization

    Args:
        omega: scale
        nbin: # of time bins
        std: standard deviation
        r: rank of incomplete Cholesky
        tol: numerical tolerance for incomplete Cholelsy
        seed: random number seed

    Returns:
        x: simulation (nbin,)
    """
    if seed is not None:
        np.random.seed(seed)
    if r is None:
        r = int(np.log(nbin))

    G = ichol_gauss(nbin, omega, r, tol)
    z = np.random.randn(r)
    x = std * G @ z
    return x


def gp_mvn(omega, std, nbin, ndim, eps=1e-8):
    """Simulate GP using multivariate normal random variable
    :param omega: 1/s^2
    :param std: standard deviation
    :param nbin: size
    :param ndim: # dimensions
    :param eps: tiny positive number
    :return: (nbin, ndim)
    """
    cov = toeplitz(np.exp(- omega * np.arange(nbin) ** 2)) + eps * np.identity(nbin)
    x = std * np.random.multivariate_normal(np.zeros(nbin), cov, ndim).T
    return x


def spikes(x, a, b, link=sexp, seed=None):
    """Simulate spike trains driven by latent processes

    Args:
        x: latent processes (T, L)
        a: coefficients of x (L, N)
        b: coefficients of regression (1 + lag*N, N)
        link: link function
        seed: random seed

    Returns:
        y: spike trains (T, N)
        h: autoregressor (T, 1 + lag*N)
        rate: firing rates (T, N)
    """

    if seed is not None:
        np.random.seed(seed)

    T, L = x.shape
    _, N = a.shape
    lag = b.shape[0] - 1

    y = np.empty((T, N), dtype=float)
    h = np.zeros((N, T, b.shape[0]), dtype=float)
    h[:, :, 0] = 1
    rate = np.empty_like(y, dtype=float)

    for t in range(T):
        eta = x[t, :] @ a + np.einsum('ij, ji -> i', h[:, t, :], b)
        rate[t, :] = link(eta)
        # truncate y to 1 if y > 1
        # equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        # y[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate[t, :]))
        y[t, :] = stats.poisson.rvs(rate[t, :]).clip(0, 1)
        if t + 1 < T and lag > 0:
            h[:, t + 1, 2:] = h[:, t, 1:lag]  # roll rightward
            h[:, t + 1, 1] = y[t, :]

    return y, h, rate


def spike(x, a, b, link=sexp, seed=None):
    """Simulate spike trains driven by latent processes

    Args:
        x: latent processes (ntrial, ntime, nlatent)
        a: coefficients of x (nlatent, nchannel)
        b: coefficients of regression (1 + lag*nchannel, nchannel)
        link: link function
        seed: random seed

    Returns:
        y: spike trains (ntrial, ntime, nchannel)
        h: autoregressor (nchannel, ntrial, ntime, 1 + lag*nchannel)
        rate: firing rates (ntrial, ntime, nchannel)
    """

    if seed is not None:
        np.random.seed(seed)

    x = np.asarray(x)
    if x.ndim < 3:
        x = np.atleast_3d(x)
        x = np.rollaxis(x, axis=-1)

    ntrial, ntime, nlatent = x.shape
    nchannel = a.shape[1]
    lag = b.shape[0] - 1

    y = np.empty((ntrial, ntime, nchannel), dtype=float)
    h = np.zeros((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    h[:, :, :, 0] = 1
    rate = np.empty_like(y, dtype=float)

    for m in range(ntrial):
        for t in range(ntime):
            eta = x[m, t, :] @ a + np.einsum('ij, ji -> i', h[:, m, t, :], b)
            rate[m, t, :] = link(eta)
            # truncate y to 1 if y > 1
            # equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
            # y[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate[t, :]))
            y[m, t, :] = stats.poisson.rvs(rate[m, t, :]).clip(0, 1)
            if t + 1 < ntime and lag > 0:
                h[:, m, t + 1, 2:] = h[:, m, t, 1:lag]  # roll rightward
                h[:, m, t + 1, 1] = y[m, t, :]

    return y, h, rate


def lfp(x, a, b, K, link=identity, seed=None):
    """Simulate LFPs driven by latent processes

    Args:
        x: latent processes (ntrial, ntime, nlatent)
        a: coefficients of x (nlatent, nchannel)
        b: coefficients of regression (1 + lag*nchannel, nchannel)
        K: noise matrix
        link: link function
        seed: random seed

    Returns:
        y: LFPs (ntrial, ntime, nchannel)
        h: autoregressor (nchannel, ntrial, ntime, 1 + lag*nchannel)
        mu: mean (ntrial, ntime, nchannel)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.asarray(x)
    if x.ndim < 3:
        x = np.atleast_3d(x)
        x = np.rollaxis(x, axis=-1)

    ntrial, ntime, nlatent = x.shape
    nchannel = a.shape[1]
    lag = b.shape[0] - 1

    y = np.empty((ntrial, ntime, nchannel), dtype=float)
    h = np.zeros((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    h[:, :, :, 0] = 1
    mu = np.empty_like(y, dtype=float)

    for m in range(ntrial):
        for t in range(ntime):
            mu[m, t, :] = link(x[m, t, :] @ a + np.einsum('ij, ji -> i', h[:, m, t, :], b))
            y[m, t, :] = multivariate_normal(mu[m, t, :], K)
            if t + 1 < ntime and lag > 0:
                h[:, m, t + 1, 2:] = h[:, m, t, 1:lag]  # roll rightward
                h[:, m, t + 1, 1] = y[m, t, :]

    return y, h, mu


# def observation(x, a, b, dist=multivariate_normal, link=identity, seed=None, *args):
#     """Simulate observations driven by latent processes
#
#     Args:
#         x: latent processes (ntrial, ntime, nlatent)
#         a: coefficients of x (nlatent, nchannel)
#         b: coefficients of regression (1 + lag*nchannel, nchannel)
#         dist: distribution
#         link: link function
#         seed: random seed
#         args: arguments for random number generation
#     Returns:
#         y: observations (ntrial, ntime, nchannel)
#         h: autoregressor (nchannel, ntrial, ntime, 1 + lag*nchannel)
#         mu: mean (ntrial, ntime, nchannel)
#     """
#     if seed is not None:
#         np.random.seed(seed)
#
#     x = np.asarray(x)
#     if x.ndim < 3:
#         x = np.atleast_3d(x)
#         x = np.rollaxis(x, axis=-1)
#
#     ntrial, ntime, nlatent = x.shape
#     nchannel = a.shape[1]
#     lag = b.shape[0] - 1
#
#     y = np.empty((ntrial, ntime, nchannel), dtype=float)
#     h = np.zeros((nchannel, ntrial, ntime, 1 + lag), dtype=float)
#     h[:, :, :, 0] = 1
#     mu = np.empty_like(y, dtype=float)
#
#     for m in range(ntrial):
#         for t in range(ntime):
#             mu[m, t, :] = link(x[m, t, :] @ a + np.einsum('ij, ji -> i', h[:, m, t, :], b))
#             y[m, t, :] = dist(mu[m, t, :], *args)
#             if t + 1 < ntime and lag > 0:
#                 h[:, m, t + 1, 2:] = h[:, m, t, 1:lag]  # roll rightward
#                 h[:, m, t + 1, 1] = y[m, t, :]
#
#     return y, h, mu


def lorenz(n, dt=0.01, s=10, r=28, b=2.667, x0=None, regular=True):
    """Lorenz attractor

    Args:
        n: the number of steps
        dt: step length, smoothness
        s:
        r:
        b:
        x0: initial values
        regular: demean, rescale

    Returns:
        3-dimenional dynamics
    """
    from numpy import empty, inf
    from numpy.linalg import norm

    def dlorenz(x, y, z):
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

    xs = empty((n, 3), dtype=float)

    # Setting initial values
    if x0 is None:
        x0 = (0., 1., 1.05)
    xs[0, :] = x0

    for i in range(n - 1):
        # Derivatives of the X, Y, Z state
        dx, dy, dz = dlorenz(xs[i, 0], xs[i, 1], xs[i, 2])
        xs[i + 1, 0] = xs[i, 0] + dx * dt
        xs[i + 1, 1] = xs[i, 1] + dy * dt
        xs[i + 1, 2] = xs[i, 2] + dz * dt

    if regular:
        xs = (xs - xs.mean(axis=0)) / norm(xs, axis=0, ord=inf)
    return xs
