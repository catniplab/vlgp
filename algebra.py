import warnings
from numpy import sqrt, exp
from numpy import sum
from numpy import zeros, ones, arange


def ichol_gauss(n, omega, r, tol=1e-6):
    """Incomplete Cholesky factorization of squared exponential covariance

    K = GG'
    Args:
        n: size of covariance matrix
        omega: inverse of 2 * squared lengthscale
        r: rank of factorization
        tol: tolerance of convergence

    Returns:
        incomplete lower trianglar matrix (n, r)
    """

    x = arange(n)
    diagG = ones(n, dtype=float)
    pvec = arange(n, dtype=int)
    i = 0
    G = zeros((n, r), dtype=float)
    while i < r and sum(diagG[i:]) > tol:
        if i > 0:
            jast = diagG[i:].argmax()
            jast += i
            # Be caseful! numpy indexing returns a view instead of a copy.
            pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            G[jast, :i + 1], G[i, :i + 1] = G[i, :i + 1].copy(), G[jast, :i + 1].copy()
        else:
            jast = 0

        G[i, i] = sqrt(diagG[jast])
        nextcol = exp(- omega * (x[pvec[i + 1:]] - x[pvec[i]]) ** 2)
        G[i + 1:, i] = (nextcol - G[i + 1:, :i].dot(G[i, :i].T)) / G[i, i]
        diagG[i + 1:] = 1 - sum((G[i + 1:, :i + 1]) ** 2, axis=1)

        i += 1
    if i == r:
        warnings.warn('Not enough rank')
    return G[pvec.argsort(), :]


def ichol(a, tol=1e-6):
    """Incomplete Cholesky factorization

    This version allows zero diagonal elements.

    Args:
        a: square matrix
        tol: tolerance of zero

    Returns:
        incomplete lower trianglar matrix
    """

    n = a.shape[0]
    diagA = a.diagonal().copy()  # Don't forget copy. diagonal() returns a read-only vector.
    pvec = arange(n, dtype=int)
    i = 0
    G = zeros((n, n), dtype=float)
    while i < n and sum(diagA[i:]) > tol:
        if i > 0:
            jast = diagA[i:].argmax()
            jast += i
            # Be caseful! numpy indexing returns a view instead of a copy.
            pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            G[jast, :i + 1], G[i, :i + 1] = G[i, :i + 1].copy(), G[jast, :i + 1].copy()
        else:
            jast = 0

        G[i, i] = sqrt(diagA[jast])
        nextcol = a[pvec[i + 1:], pvec[i]]
        G[i + 1:, i] = (nextcol - G[i + 1:, :i].dot(G[i, :i].T)) / G[i, i]
        diagA[i + 1:] = 1 - sum((G[i + 1:, :i + 1]) ** 2, axis=1)

        i += 1
    return G[pvec.argsort(), :]
