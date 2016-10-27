"""
Tool functions
"""
import warnings

import h5py
import numpy as np
from numpy import exp, column_stack, roll
from numpy import zeros, ones, diag, arange, eye, asarray, atleast_3d, rollaxis
from scipy.linalg import svd, lstsq, toeplitz, solve
from scipy.ndimage.filters import gaussian_filter1d

from .math import ichol_gauss


def makeregressor(obs, p):
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
            regressor[t, 1:] = obs[t - p:t, :].flatten()  # by row
        else:
            regressor[t, 1 + (p - t) * N:] = obs[:t, :].flatten()
    return regressor


def sqexpcov(n, w, var=1.0):
    """Construct square exponential covariance matrix

    Args:
        n: size of the matrix
        w: scale
        var: variance

    Returns:
        covariance
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


def history(obs, lag):
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


def rotate(obj, ref):
    """Rotation

    Args:
        obj:
        ref:

    Returns:

    """
    return obj @ lstsq(obj, ref)[0]


def add_constant(x):
    """Add an all-one column to matrix

    Args:
        x: matrix

    Returns:

    """
    x = asarray(x)
    x = column_stack((x, ones((x.shape[0], 1))))
    return roll(x, 1, 1)


def lagmat(x, lag):
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
        mat[lag - k:nrow + lag - k, ncol * (lag - k):ncol * (lag - k + 1)] = x
    startrow = 0
    stoprow = nrow + lag - k

    return mat[startrow:stoprow, ncol:]


def save(obj, fname, warning=False):
    """
    Save inference object in HDF5

    Parameters
    ----------
    obj: dict
        inference
    fname: string
        absolute path and filename
    warning: bool
        print warning if any field in obj is not supported by HDF5
    """
    with h5py.File(fname, 'w') as hf:
        for k, v in obj.items():
            try:
                hf.create_dataset(k, data=v, compression="gzip")
            except TypeError:
                msg = 'Discard unsupported type ({})'.format(k)
                if warning:
                    warnings.warn(msg)


def load(fname):
    with h5py.File(fname, 'r') as hf:
        obj = {k: np.asarray(v) for k, v in hf.items()}
    return obj


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
        L, _, M = svd(A.T @ (n * B ** 3 - gamma * B @ diag(sum(B ** 2, axis=0))),
                      full_matrices=False)  # the sum of each column
        T = L @ M
        if norm(T - eye(m)) < rtol:
            T, _ = qr(randn(m, m));
            B = A @ T
        s = 0
        for k in range(maxit):
            s_old = s
            L, s, M = svd(A.T @ (n * B ** 3 - gamma * B @ diag(sum(B ** 2, axis=0))), full_matrices=False)
            T = L @ M
            s = sum(s)
            B = A @ T
            if (s - s_old) < rtol * s:
                converged = True
                break

    if not converged:
        warnings.warn('iteration limit')

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


def trial_slices(trial_lengths: list):
    from numpy import cumsum, s_
    ntrial = len(trial_lengths)
    endpoints = [0] + trial_lengths
    endpoints = cumsum(endpoints)
    slices = []
    for i in range(ntrial):
        slices.append(s_[endpoints[i]:endpoints[i+1]])
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
    return np.concatenate([np.stack([add_constant(lagmat(col, lag)) for col in trial.T]) for trial in y], axis=1)


def sparse_prior(sigma, omega, trial_lengths, rank):
    # [diagonal(G1, G2, ..., Gq)]
    from scipy import sparse
    return [sparse.block_diag([s * ichol_gauss(l, w, rank) for s, w in zip(sigma, omega)]) for l in trial_lengths]


def regmat(y, x=None, lag=0):
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
    y_ndim = automat.shape[0]
    return np.concatenate([automat, np.stack([big_x] * y_ndim)], axis=2)


def smooth_1d(x, sigma=10):
    assert x.ndim == 1
    y = gaussian_filter1d(x, sigma=sigma, mode='constant', cval=0.0)
    return y


def smooth(x, sigma=10):
    return np.stack([smooth_1d(row, sigma) for row in x])
