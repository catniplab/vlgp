import numpy as np

from numpy import pi, exp, sqrt, log2, ceil, fft, einsum
from numpy import zeros, arange, empty, ones, roll, empty_like
from numpy.random import random, multivariate_normal
from scipy import stats

from link import *

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

    y = empty((T, N), dtype=float)
    h = zeros((N, T, b.shape[0]), dtype=float)
    h[:, :, 0] = 1
    rate = empty_like(y, dtype=float)

    for t in range(T):
        eta = x[t, :].dot(a) + einsum('ij, ji -> i', h[:, t, :], b)
        rate[t, :] = link(eta)
        # truncate y to 1 if y > 1
        # equivalent to Bernoulli P(1) = (1 - e^-(lam_t))
        # y[t, :] = stats.bernoulli.rvs(1.0 - exp(-rate[t, :]))
        y[t, :] = stats.poisson.rvs(rate[t, :]).clip(0, 1)
        if t + 1 < T and lag > 0:
            h[:, t + 1, 2:] = h[:, t, 1:lag]  # roll rightward
            h[:, t + 1, 1] = y[t, :]

    return y, h, rate


def lfp(x, a, b, K, link=identity, seed=None):
    """Simulate LFPs driven by latent processes

    Args:
        x: latent processes (T, L)
        a: coefficients of x (L, N)
        b: coefficients of regression (1 + lag*N, N)
        K: noise
        link: link function
        seed: random seed

    Returns:
        y: LFPs
        h: autoregressor
        mu: mean
    """
    if seed is not None:
        np.random.seed(seed)

    T, L = x.shape
    _, N = a.shape
    lag = b.shape[0] - 1

    y = empty((T, N), dtype=float)
    mu = empty_like(y, dtype=float)
    h = zeros((N, T, b.shape[0]), dtype=float)
    h[:, :, 0] = 1

    for t in range(T):
        mu[t, :] = identity(x[t, :].dot(a) + einsum('ij, ji -> i', h[:, t, :], b))
        y[t, :] = multivariate_normal(mu[t, :], K)
        if t + 1 < T and lag > 0:
            h[:, t + 1, 2:] = h[:, t, 1:lag]  # roll rightward
            h[:, t + 1, 1] = y[t, :]

    return y, h, mu


def lorenz(n, dt=0.01, s=10, r=28, b=2.667, x0=None, constraint=True):
    """Lorenz attractor

    Args:
        n: the number of steps
        dt: step length, smoothness
        s:
        r:
        b:
        x0: initial values
        constraint: demean, rescale

    Returns:

    """
    from numpy import empty, inf
    from numpy.linalg import norm

    def dlorenz(x, y, z):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
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

    if constraint:
        xs = (xs - xs.mean(axis=0)) / norm(xs, axis=0, ord=inf)
    return xs
