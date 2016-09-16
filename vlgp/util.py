"""
Tool functions
"""
import warnings

import h5py
import numpy as np
from numpy import exp, column_stack, roll, sum, dot
from numpy import zeros, ones, diag, arange, eye, asarray, atleast_3d, rollaxis
from scipy.linalg import svd, lstsq, toeplitz, solve


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


def regmat(y, lag=0):
    """Autoregression matrices

    Args:
        y: observation
        lag: lag

    Returns:
        autoregression matrices (nchannel, ntrial, ntime, 1 + lag)
    """

    y = asarray(y)
    if y.ndim < 3:
        y = atleast_3d(y)
        y = rollaxis(y, axis=-1)
    ntrial, ntime, nchannel = y.shape
    h = zeros((nchannel, ntrial, ntime, 1 + lag))
    for n in range(nchannel):
        for m in range(ntrial):
            h[n, m, :] = add_constant(lagmat(y[m, :, n], lag=lag))
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
        obj = {k: np.array(v) for k, v in hf.items()}
    return obj
