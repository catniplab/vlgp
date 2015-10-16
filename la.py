import numpy as np
import warnings


def ichol_gauss(n, omega, k, tol=1e-8):
    """
    Incomplete Cholesky decomposition for squared exponential covariance
    :param n: size of covariance matrix (n, n)
    :param omega: inverse of 2 * squared lengthscale
    :param k: number of columns of decomposition
    :return: (n, m) matrix
    """
    x = np.arange(n)
    # x = np.linspace(0, 1, n)
    diagG = np.ones(n, dtype=float)
    pvec = np.arange(n, dtype=int)
    i = 0
    g = np.zeros((n, k), dtype=float)
    vtmp = np.zeros(k, dtype=float)
    while i < k and np.sum(diagG[i:]) > tol:
        if i > 0:
            jast = np.argmax(diagG[i:])
            jast += i
            # Be caseful especially when you do assignment! numpy indexing returns a view instead of a copy.
            pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            # vtmp[:i + 1] = g[i, :i + 1]
            # g[i, :i + 1] = g[jast, :i + 1]
            # g[jast, :i + 1] = vtmp[:i + 1]
            g[jast, :i + 1], g[i, :i + 1] = g[i, :i + 1].copy(), g[jast, :i + 1].copy()
        else:
            jast = 0

        g[i, i] = np.sqrt(diagG[jast])
        newAcol = np.exp(- omega * (x[pvec[i + 1:]] - x[pvec[i]]) ** 2)
        g[i + 1:, i] = (newAcol - np.dot(g[i + 1:, :i], g[i, :i].T)) / g[i, i]
        diagG[i + 1:] = 1 - np.sum((g[i + 1:, :i + 1]) ** 2, axis=1)

        i += 1
    if i == k:
        warnings.warn('Not enough ranks')
    return g[np.argsort(pvec), :]


def ichol(a):
    """
    Incomplete Cholesky factorization
    Note the diagonal elements cannot be too small to divide. Use ichol2.
    :param a: square matrix
    :return: lower triangle
    """
    n = a.shape[0]
    for k in range(n):
        a[k, k] = np.sqrt(a[k, k])
        for i in range(k + 1, n):
            if a[i, k] != 0:
                a[i, k] /= a[k, k]

        for j in range(k + 1, n):
            for i in range(j, n):
                if a[i, j] != 0:
                    a[i, j] -= a[i, k] * a[j, k]

    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = 0

    return a


def ichol2(a, tol=1e-8):
    """
    Incomplete Cholesky factorization
    This version allows too small diagonal elements.
    :param a: square matrix
    :param tol: numerical tolerance
    :return: lower triangle
    """
    n = a.shape[0]
    diagA = a.diagonal().copy()  # Don't forget copy. diagonal() returns a read-only vector.
    pvec = np.arange(n, dtype=int)
    i = 0
    g = np.zeros((n, n), dtype=float)
    vtmp = np.zeros_like(pvec)
    while i < n and np.sum(diagA[i:]) > tol:
        if i > 0:
            jast = np.argmax(diagA[i:])
            jast += i
            # pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            # g[jast, :i + 1], g[i, :i + 1] = g[i, :i + 1].copy(), g[jast, :i + 1].copy()
            # stmp = pvec[jast]
            # pvec[jast] = pvec[i]
            # pvec[i] = stmp
            pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            vtmp[:i + 1] = g[i, :i + 1]
            g[i, :i + 1] = g[jast, :i + 1]
            g[jast, :i + 1] = vtmp[:i + 1]
        else:
            jast = 0

        g[i, i] = np.sqrt(diagA[jast])
        newAcol = a[pvec[i + 1:], pvec[i]]
        g[i + 1:, i] = (newAcol - np.dot(g[i + 1:, :i], g[i, :i].T)) / g[i, i]
        diagA[i + 1:] = 1 - np.sum((g[i + 1:, :i + 1]) ** 2, axis=1)

        i += 1
    return g[np.argsort(pvec), :]
