import timeit

import numpy as np
from numpy import empty, full, tile
from numpy import finfo, PINF, inf
from numpy import identity, diag, einsum, inner, trace, sum, mean, var, abs, sqrt
from scipy import linalg

from util import selfhistory


LB = -20
UB = 20


def vfromw(w, chol):
    """Construct temporal slice of V from W

    Args:
        w: diagonals of W (T, L)
        chol: cholesky factorizations of prior covariances (L, T, r)

    Returns:
        v: diagonals of V (T, L)
    """
    L, T, r = chol.shape
    eyer = identity(r)
    v = empty((T, L), dtype=float)
    for l in range(L):
        G = chol[l, :]
        GTWG = G.T.dot(w[l] * G)
        v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True)))),
                      axis=1)
    return v


def elbo(y, h, chol, m, w, v, a, b, vhat):
    """Evidence Lower BOund

    Args:
        y: observations (T, N)
        h: autocorrelation regressor (N, T, 1 + p)
        chol: cholesky factorizations of prior covariances (L, T, r)
        m: posterior mean (T, L)
        w: diagonals of W (T, L)
        v: temporal slice of V (T, L)
        a: latent coefficients (L, N)
        b: autocorrelation coefficients (1 + p, N)
        vhat: MLE of sample variances

    Returns:

    """
    L, T, r = chol.shape
    eyer = identity(r)

    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)

    lb = - 0.5 * sum(((y - eta) ** 2 + v.dot(a ** 2)) / vhat)

    for l in range(L):
        G = chol[l, :]
        GTWG = G.T.dot(w[l] * G)
        A = GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True))
        m_div_G = linalg.lstsq(G, m[:, l])[0]
        tr = T - trace(GTWG) + trace(A)
        lndet = np.linalg.slogdet(eyer - GTWG + A)[1]

        lb += -0.5 * inner(m_div_G, m_div_G) - 0.5 * tr + 0.5 * lndet

    return lb


def train(y, p, chol, m0=None, a0=None, b0=None, abest=True, niter=50, tol=1e-5, verbose=True):
    """Variational Bayesian

    Args:
        y: observations (T, N), continuous
        p: order of autocorrelation
        chol: cholesky factorizations of prior covariances (L, T, r)
        m0: initial posterior mean (T, L)
        a0: initial latent coefficients (L, N)
        b0: initial autocorrelation coefficients (1 + p, N)
        niter: max number of iterations
        tol: relative tolerance of convergence
        verbose: detailed output

    Returns:
        lbound: ELBO of all iterations
        m: posterior mean (T, L)
        lv: L matrices in factorization of V = LL' (L, T, r)
        a: latent coefficients (L, N)
        b: autocorrelation coefficients (1 + p, N)
        elapsed: running time
        converged: whether the algorithm converged within iteration limit
    """
    L, T, r = chol.shape
    N = y.shape[1]
    eyek = identity(r)

    y0 = mean(y, axis=0)
    h = selfhistory(y, p, y0)

    if m0 is None:
        m0 = tile(mean(y, axis=1), (1, L))
        m0 -= mean(m0, axis=0)

    if a0 is None:
        a0 = linalg.lstsq(m0, y)[0]

    if b0 is None:
        b0 = empty((1 + p, N), dtype=float)
        for n in range(N):
            b0[:, n] = linalg.lstsq(h[n, :], y[:, n])[0]

    a = a0
    b = b0
    m = m0

    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    vhat = var(y - eta, axis=0, ddof=0)
    w = (1 / vhat).dot(a.T ** 2)
    v = vfromw(w, chol)

    lbound = full(niter, fill_value=finfo(float).min, dtype=float)
    lbound[0] = elbo(y, h, chol, m, w, v, a, b, vhat)

    chosen = full(L, fill_value=True, dtype=bool)  #

    converged = False
    i = 1
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and i < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(i))

        chosen.fill(True)
        for l in range(L):
            chosen[l] = False
            z = (y - einsum('ijk, ki->ji', h, b) - m[:, chosen].dot(a[chosen, :])).dot(a[l, :] / vhat)
            G = chol[l, :]
            GTWG = G.T.dot(w[l] * G)
            m[:, l] = (G - G.dot(GTWG) + G.dot(GTWG).dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))).dot(G.T.dot(z))
            if abest:
                m[:, l] -= mean(m[:, l])
                m[:, l] /= linalg.norm(m[:, l], ord=inf)
            # v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
            #                  axis=1)
        if abest:
            for n in range(N):
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                a[:, n] = linalg.solve(m.T.dot(m) + diag(sum(v, axis=0)), m.T.dot(y[:, n] - h[n, :].dot(b[:, n])),
                                       sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, n] = linalg.solve(h[n, :].T.dot(h[n, :]), h[n, :].T.dot(y[:, n] - m.dot(a[:, n])), sym_pos=True)

        eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
        vhat = var(y - eta, axis=0, ddof=0)
        w = (1 / vhat).dot(a.T ** 2)
        v = vfromw(w, chol)

        lbound[i] = elbo(y, h, chol, m, w, v, a, b, vhat)

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'.format(lbound[i], lbound[i] - lbound[i - 1],
                                            timeit.default_timer() - iter_start))

        # check convergence
        if abs(lbound[i] - lbound[i - 1]) < tol * abs(lbound[i - 1]):
            converged = True

        i += 1
    stop = timeit.default_timer()

    lv = empty((L, T, r), dtype=float)  # V = LL'
    for l in range(L):
        G = chol[l, :]
        GTWG = G.T.dot(w[l] * G)

        A = eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))  # A should be pd but numerically not
        eigval, eigvec = linalg.eigh(A)
        eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
        lv[l, :] = G.dot(eigvec.dot(diag(sqrt(eigval))))

    return lbound[:i], m, lv, a, b, stop - start, converged
