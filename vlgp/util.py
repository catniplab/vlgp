"""
Tool functions
"""
import functools
import logging
import numbers
import pathlib
import warnings
from typing import List, Optional, Callable

import h5py
import numpy as np
from numpy import exp, column_stack, roll
from numpy import zeros, ones, diag, arange, eye, asarray
from scipy.linalg import svd, lstsq, toeplitz, solve
from scipy.ndimage.filters import gaussian_filter1d

from .math import ichol_gauss

logger = logging.getLogger(__name__)


def makeregressor(obs, p: int):
    """Construct full regressive matrix

    Args:
        obs: observations (T, N)
        p: order of auto/cross-regression

    Returns:
        full design matrix (T, 1 + p*N)
    """
    T, N = obs.shape
    regressor = ones((T, 1 + p * N), dtype=float)
    for t in range(T):
        if t - p >= 0:
            regressor[t, 1:] = obs[t - p : t, :].flatten()  # by row
        else:
            regressor[t, 1 + (p - t) * N :] = obs[:t, :].flatten()
    return regressor


def sqexpcov(n: int, w: float, var: float = 1.0):
    """Construct square exponential covariance matrix

    Args:
        n: size of the matrix
        w: scale
        var: variance

    Returns:
    """

    # i, j = meshgrid(arange(n), arange(n))
    # return var * exp(- w * (i - j) ** 2)
    return var * exp(-w * toeplitz(arange(n)))


def promax(x, m=4):
    """
    function (x, m = 4)
    {
        if (ncol(x) < 2)
            return(x)
        dn <- dimnames(x)
        xx <- varimax(x)
        x <- xx$loadings
        Q <- x * abs(x)^(m - 1)
        U <- lm.fit(x, Q)$coefficients
        d <- diag(solve(t(U) %*% U))
        U <- U %*% diag(sqrt(d))
        dimnames(U) <- NULL
        z <- x %*% U
        U <- xx$rotmat %*% U
        dimnames(z) <- dn
        class(z) <- "loadings"
        list(loadings = z, rotmat = U)
    }
    """
    if x.shape[1] < 2:
        return x
    xT, TT = varimax(x)
    Q = xT * np.abs(xT) * (m - 1)
    U = lstsq(xT, Q)[0]
    d = diag(solve(np.inner(U, U), eye(U.shape[1])))
    U = U @ diag(np.sqrt(d))
    z = xT @ U
    return z, U


def history(obs, lag: int):
    """Construct autoregressive matrices

    Args:
        obs: observations (ntime, nchannel)
        lag: order of autoregression

    Returns:
        autoregression matrices (nchannel, ntime, 1 + lag)
    """

    ntime, nchannel = obs.shape
    h = zeros((nchannel, ntime, 1 + lag), dtype=float)

    for n in range(nchannel):
        h[n, :] = add_constant(lagmat(obs[:, n], lag=lag))

    return h


def rotate(x, y):
    """Rotation

    Args:
        x:
        y:

    Returns:

    """
    return x @ lstsq(x, y)[0]


def add_constant(x):
    """Add an all-one column to matrix

    Args:
        x: matrix

    Returns:

    """
    x = asarray(x)
    x = column_stack((x, ones((x.shape[0], 1))))
    return roll(x, 1, 1)


def lagmat(x, lag: int):
    """Make autoregression matrix

    Args:
        x: vector
        lag:

    Returns:

    """
    x = asarray(x)
    if x.ndim < 2:
        x = x[..., None]
    nrow, ncol = x.shape
    if lag >= nrow:
        raise ValueError("lag should be < nrow")
    mat = zeros((nrow + lag, ncol * (lag + 1)))
    for k in range(0, int(lag + 1)):
        mat[lag - k : nrow + lag - k, ncol * (lag - k) : ncol * (lag - k + 1)] = x
    startrow = 0
    stoprow = nrow + lag - k

    return mat[startrow:stoprow, ncol:]


# def save(obj, fname: str):
#     """
#     Save inference object in HDF5
#
#     Parameters
#     ----------
#     obj: dict
#         inference
#     fname: string
#         absolute path and filename
#     """
#     with h5py.File(fname, 'w') as fout:
#         dict_to_hdf5(obj, fout)


# def load(fname: str):
#     with h5py.File(fname, 'r') as fin:
#         obj = hdf5_to_dict(fin)
#     return obj


def save(result, path=None, code="npy"):
    """Save *ANYTHING*"""
    if path is None:
        path = result["path"]
    else:
        result["path"] = path
    path = pathlib.Path(path)

    if code == "h5":
        path = path.with_suffix(".h5")
        with h5py.File(path, "w") as fout:
            dict_to_hdf5(result, fout)
    elif code == "npy":
        path = path.with_suffix(".npy")
        np.save(path, result)
    elif code == "npz":
        path = path.with_suffix(".npz")
        np.savez(path, **result)


def load(path):
    """Load result from file"""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path.as_posix())

    if path.suffix == ".h5":
        with h5py.File(path.as_posix(), "r") as fin:
            rez = hdf5_to_dict(fin)
    elif path.suffix == ".npy":
        rez = np.load(path)
        rez = rez[()]
    elif path.suffix == ".npz":
        rez = np.load(path)
        rez = {**rez}
    else:
        raise NotImplementedError("unknown file type {}".format(path.suffix))

    return rez


def orthomax(A, gamma=1.0, normalize=True, rtol=1e-8, maxit=250):
    """Orthogonal rotation of FA or PCA loadings"""
    from scipy.linalg import svd, norm, qr
    from numpy import sum, eye, sqrt
    from numpy.random import randn

    A = A.copy()
    n, m = A.shape
    if normalize:
        h = sqrt(np.sum(A ** 2, axis=1, keepdims=True))
        A /= h

    T = eye(m)
    B = A @ T

    converged = False
    if 0 <= gamma <= 1:
        L, _, M = svd(
            A.T @ (n * B ** 3 - gamma * B @ diag(sum(B ** 2, axis=0))),
            full_matrices=False,
        )  # the sum of each column
        T = L @ M
        if norm(T - eye(m)) < rtol:
            T, _ = qr(randn(m, m))
            B = A @ T
        s = 0
        for k in range(maxit):
            s_old = s
            L, s, M = svd(
                A.T @ (n * B ** 3 - gamma * B @ diag(sum(B ** 2, axis=0))),
                full_matrices=False,
            )
            T = L @ M
            s = sum(s)
            B = A @ T
            if (s - s_old) < rtol * s:
                converged = True
                break

    if not converged:
        warnings.warn("iteration limit")

    if normalize:
        B *= h

    return B, T


def varimax(x, normalize=True, tol=1e-5, niter=1000):
    """
    Varimax rotation stolen from R

    function (x, normalize = TRUE, eps = 1e-05)
    {
        nc <- ncol(x)
        if (nc < 2)
            return(x)
        if (normalize) {
            sc <- sqrt(drop(apply(x, 1L, function(x) sum(x^2))))
            x <- x/sc
        }
        p <- nrow(x)
        TT <- diag(nc)
        d <- 0
        for (i in 1L:1000L) {
            z <- x %*% TT
            B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
            sB <- La.svd(B)
            TT <- sB$u %*% sB$vt
            dpast <- d
            d <- sum(sB$d)
            if (d < dpast * (1 + eps))
                break
        }
        z <- x %*% TT
        if (normalize)
            z <- z * sc
        dimnames(z) <- dimnames(x)
        class(z) <- "loadings"
        list(loadings = z, rotmat = TT)
    }
    """
    x = x.copy()
    p, nc = x.shape

    if nc < 2:
        return x

    if normalize:
        sc = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))  # ???
        x /= sc

    TT = eye(nc)
    d = 0
    for i in range(niter):
        z = x @ TT
        B = x.T @ (z ** 3 - z @ np.diag(np.sum(z ** 2, axis=0)) / p)
        U, s, Vh = svd(B, full_matrices=False)
        TT = U @ Vh
        dpast = d
        d = np.sum(s)
        if d < dpast * (1 + tol):
            break

    z = x @ TT
    if normalize:
        z *= sc
    return z, TT


def trial_slices(trial_lengths: List[int]):
    from numpy import cumsum, s_

    ntrial = len(trial_lengths)
    endpoints = [0] + trial_lengths
    endpoints = cumsum(endpoints)
    slices = []
    for i in range(ntrial):
        slices.append(s_[endpoints[i] : endpoints[i + 1]])
    return slices


def auto(y, lag):
    """

    Parameters
    ----------
    y : list
        [array[time, y_ndim]]
    lag :

    Returns
    -------
    array[y_ndim, time, lag + 1]
    """
    assert len(y) > 0
    return np.concatenate(
        [np.stack([add_constant(lagmat(col, lag)) for col in trial.T]) for trial in y],
        axis=1,
    )


def sparse_prior(sigma, omega, trial_lengths, rank):
    # [diagonal(G1, G2, ..., Gq)]
    from scipy import sparse

    return [
        sparse.block_diag([s * ichol_gauss(l, w, rank) for s, w in zip(sigma, omega)])
        for l in trial_lengths
    ]


def regmat(y, x: Optional[list], lag=0):
    """

    Parameters
    ----------
    y : list
        observation
    x : list
        external variables
        [array(time, x_ndim)]
    lag : int

    Returns
    -------

    """
    automat = auto(y, lag)
    big_x = np.concatenate(x, axis=0)  # along time
    y_dim = automat.shape[0]
    return np.concatenate([automat, np.stack([big_x] * y_dim)], axis=2)


def smooth_1d(x, sigma=10):
    assert x.ndim == 1
    y = gaussian_filter1d(x, sigma=sigma, mode="constant", cval=0.0)
    return y


def smooth(x, sigma=10):
    return np.stack([smooth_1d(row, sigma) for row in x.T]).T


def dict_to_hdf5(d: dict, hdf):
    for key, value in d.items():
        if isinstance(value, dict):
            group = hdf.create_group(key)
            dict_to_hdf5(value, group)
        else:
            try:
                if isinstance(value, np.ndarray):
                    hdf.create_dataset(key, data=value, compression="gzip")
                else:
                    hdf.create_dataset(key, data=value)
            finally:
                pass


def hdf5_to_dict(hdf):
    d = dict()
    for key, value in hdf.items():
        if isinstance(value, h5py.Group):
            d[key] = hdf5_to_dict(value)
        else:
            d[key] = value[()]
    return d


def log(f: Callable):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        logger.info("{:s} is called".format(f.__name__))
        return f(*args, **kwargs)

    return wrapper


def transform(timescale, dt):
    """
    Transform timescale to omega

    Parameters
    ----------
    timescale : float or array
    dt : float

    Returns
    -------
    float
    """

    return 0.5 * (dt / timescale) ** 2


def clip(a, lbound, ubound=None):
    """Clip an array by given bounds in place"""
    if ubound is None:
        assert lbound > 0
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(a, lbound, ubound, out=a)


def cut_trials(trials, params, config):
    """Cut all trials"""
    window = config["window"]
    if window and window is not None:
        return np.concatenate(
            [cut_trial(trial, window) for trial in trials]
        )  # concatenate segments
    else:
        return trials


def cut_trial(trial, window: int):
    """Cut a trial into small segments"""
    import math

    y = trial["y"]
    x = trial["x"]
    mu = trial["mu"]
    w = trial["w"]
    v = trial["v"]

    length = y.shape[0]

    # allow overlapping segments if the trial length is not a multiplier of window
    # random sample the segment starting points
    num_segments = math.ceil(length / window)
    overlap = num_segments * window - length  # number of overlapping segments
    start = np.cumsum(np.full(num_segments, fill_value=window, dtype=int)) - window
    offset = np.cumsum(
        np.append(
            [0],
            np.random.multinomial(
                overlap, np.ones(num_segments - 1) / (num_segments - 1)
            ),
        )
    )
    start -= offset
    slices = [np.s_[s : s + window] for s in start]
    segments = [
        {"y": y[s, :], "x": x[s, ...], "mu": mu[s, :], "w": w[s, :], "v": v[s, :]}
        for s in slices
    ]
    return segments


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance"""
    if seed is None or seed is np.random:
        return np.random.get_state()
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)
