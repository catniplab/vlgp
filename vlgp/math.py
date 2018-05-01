"""
Math functions
"""
import warnings

import numpy as np
from scipy import linalg
from scipy.linalg import svd


def rectify(x):
    """
    rectangular linear link

    Args:
        x: linear predictor
    """
    return x.clip(0, np.inf)


def trunc_exp(x, bound=10):
    """
    Truncated exp

    Parameters
    ----------
    x : ndarray
    bound : double
        upper bound of x
    Returns
    -------
    ndarray
        exp(min(x, ubound))
    """
    return np.exp(np.minimum(x, bound))


def lexp(x, c=0):
    """Linearized exp"""
    return np.exp(x) if x < c else np.exp(c) * (1 - c + x)


def identity(x):
    """
    Identity function

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    ndarray
    """
    return x


def log1exp(x):
    """
    Function: log(1 + exp(x))

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    ndarray
    """
    return np.log1p(np.exp(x))


def ichol_gauss_old(n, omega, r, tol=1e-6):
    """
    Incomplete Cholesky factorization of squared exponential covariance matrix
    A = GG' + E

    Parameters
    ----------
    n : int
        size of matrix
    omega : double
        1 / (2 * timescale^2)
    r : int
        rank
    tol : double
        numerical tolerance

    Returns
    -------
    ndarray
        (n, r) matrix
    """
    x = np.arange(n)
    diag = np.ones(n, dtype=float)
    pvec = np.arange(n, dtype=int)
    i = 0
    G = np.zeros((n, r), dtype=float)  # preallocation
    while i < r and np.sum(diag[i:]) > tol * n:
        if i > 0:
            jast = diag[i:].argmax()
            jast += i
            # Be caseful! numpy indexing returns a view instead of a copy.
            pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            G[jast, :i + 1], G[i, :i + 1] = G[i, :i + 1].copy(), G[jast, :i + 1].copy()
        else:
            jast = 0

        G[i, i] = np.sqrt(diag[jast])
        nextcol = np.exp(- omega * (x[pvec[i + 1:]] - x[pvec[i]]) ** 2)
        G[i + 1:, i] = (nextcol - G[i + 1:, :i] @ G[i, :i].T) / G[i, i]
        diag[i + 1:] = 1 - np.sum((G[i + 1:, :i + 1]) ** 2, axis=1)

        i += 1

    return G[pvec.argsort(), :]


def ichol_gauss(n, omega, r, tol=1e-6):
    """
    Incomplete Cholesky factorization of squared exponential covariance matrix
    A = GG' + E

    Parameters
    ----------
    n : int
        size of matrix
    omega : double
        1 / (2 * timescale^2)
    r : int
        rank
    tol : double
        numerical tolerance

    Returns
    -------
    ndarray
        (n, r) matrix
    """
    # TODO: more effifient algorithm
    x = np.arange(n)
    diag = np.ones(n, dtype=float)
    pvec = np.arange(n, dtype=int)  # pivot
    i = 0
    G = np.zeros((n, r), dtype=float)  # preallocation
    while i < r and np.sum(diag[i:]) > tol * n:
        if i > 0:
            jast = diag[i:].argmax()
            jast += i
            pvec[[i, jast]] = pvec[[jast, i]]
            G[[i, jast], :i + 1] = G[[jast, i], :i + 1]  # avoid copy
        else:
            jast = 0

        G[i, i] = np.sqrt(diag[jast])
        nextcol = np.exp(- omega * (x[pvec[i + 1:]] - x[pvec[i]]) ** 2)
        G[i + 1:, i] = (nextcol - np.dot(G[i + 1:, :i], G[i, :i])) / G[i, i]
        diag[i + 1:] = 1 - np.sum(np.square(G[i + 1:, :i + 1]), axis=1)

        i += 1

    return G[pvec.argsort(), :]


def ichol(a, tol=1e-6):
    """
    Incomplete Cholesky factorization

    This version allows zero diagonal elements.
    The direct implementation of the version in wikipedia,
    https://en.wikipedia.org/wiki/Incomplete_Cholesky_factorization,
    encounters divided by zero.

    Args:
        a: square matrix
        tol: tolerance of zero

    Returns:
        incomplete lower trianglar matrix
    """

    n = a.shape[0]
    diag = a.diagonal().copy()  # Don't forget copy. diagonal() returns a read-only vector.
    pvec = np.arange(n, dtype=int)
    i = 0
    G = np.zeros((n, n), dtype=float)
    while i < n and np.sum(diag[i:]) > tol:
        if i > 0:
            jast = diag[i:].argmax()
            jast += i
            # Be caseful! numpy indexing returns a view instead of a copy.
            pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            G[jast, :i + 1], G[i, :i + 1] = G[i, :i + 1].copy(), G[jast, :i + 1].copy()
        else:
            jast = 0

        G[i, i] = np.sqrt(diag[jast])
        nextcol = a[pvec[i + 1:], pvec[i]]
        G[i + 1:, i] = (nextcol - G[i + 1:, :i] @ G[i, :i].T) / G[i, i]
        diag[i + 1:] = 1 - np.sum((G[i + 1:, :i + 1]) ** 2, axis=1)

        i += 1
    return G[pvec.argsort(), :i]


def subspace(a, b, deg=True):
    """
    Angle between two subspaces specified by the columns of a and b
    Ported from MATLAB 'subspace' function

    Parameters
    ----------
    a : matrix
    b : matrix
    deg : bool
        return degree or radian

    Returns
    -------
    double
        angle
    """
    warnings.warn("Deprecated. Use scipy.linalg.subspace_angles instead.", FutureWarning)
    oa = linalg.orth(a)
    ob = linalg.orth(b)
    if oa.shape[1] < ob.shape[1]:
        oa, ob = ob.copy(), oa.copy()
    ob -= oa @ (oa.T @ ob)
    rad = np.arcsin(min(1, linalg.norm(ob, ord=2)))
    return np.degrees(rad) if deg else rad


def orth(x, a):
    """
    Orthogonalize the rows of the loading matrix and apply the corresponding linear transform to the latent variables.

    Args:
        x: latent variables
        a: loading matrix

    Returns:

    """
    U, s, Vh = svd(a, full_matrices=False)  # scipy version
    a_orth = Vh
    x_orth = x @ a @ Vh.T
    return x_orth, a_orth


def diagadd(m, v):
    """Add a vector to the diagonal of a matrix"""
    np.fill_diagonal(m, m.diagonal() + v)
