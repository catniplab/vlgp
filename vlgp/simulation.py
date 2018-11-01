"""
Functions of simulation
"""
import numpy as np
from numpy.random import multivariate_normal
from scipy import stats

from .math import trunc_exp, identity


def spike(x, a, b, link=trunc_exp, seed=None):
    """Simulate spike trains

        firing rate = exp(latent process . loading matrix + spike history * history filter + bias)
        where . represents matrix multiplication and * represents convolution

    :param x: latent process
    :type x: ndarray
    :param a: loading matrix (right)
    :type a: ndarray
    :param b: history filter and bias
    :type b: ndarray
    :param link: link function
    :type link: callable
    :param seed: random number seed
    :type seed: optional[int]
    :return: spike train, spike history, firing rate
    :rtype: ndarray, ndarray, ndarray
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
            eta = x[m, t, :] @ a + np.einsum("ij, ji -> i", h[:, m, t, :], b)
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
            mu[m, t, :] = link(
                x[m, t, :] @ a + np.einsum("ij, ji -> i", h[:, m, t, :], b)
            )
            y[m, t, :] = multivariate_normal(mu[m, t, :], K)
            if t + 1 < ntime and lag > 0:
                h[:, m, t + 1, 2:] = h[:, m, t, 1:lag]  # roll rightward
                h[:, m, t + 1, 1] = y[m, t, :]

    return y, h, mu


def lorenz(n, dt=0.01, s=10, r=28, b=2.667, x0=None, normalized=False):
    """Generate a trajectory of Lorenz attractor

    :param n: length
    :type n: int
    :param dt: time step
    :type dt: float
    :param s: parameter
    :param r: parameter
    :param b: parameter
    :param x0: initial state
    :type x0: (float, float, float)
    :param normalized: z-score
    :type normalized: bool
    :return: a trajectory
    :rtype: ndarray
    """
    from numpy import empty, inf
    from numpy.linalg import norm

    def dot(x, y, z):
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

    xs = empty((n, 3), dtype=float)

    # Setting initial values
    if x0 is None:
        x0 = (0.0, 1.0, 1.05)
    xs[0, :] = x0

    for i in range(n - 1):
        # Derivatives of the X, Y, Z state
        dx, dy, dz = dot(xs[i, 0], xs[i, 1], xs[i, 2])
        xs[i + 1, 0] = xs[i, 0] + dx * dt
        xs[i + 1, 1] = xs[i, 1] + dy * dt
        xs[i + 1, 2] = xs[i, 2] + dz * dt

    if normalized:
        xs = (xs - xs.mean(axis=0)) / norm(xs, axis=0, ord=inf)

    return xs
