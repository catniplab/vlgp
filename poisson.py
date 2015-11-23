import timeit

import numpy as np
from numpy import empty, empty_like, full, zeros, zeros_like, newaxis, tile
from numpy import identity, diag, einsum, inner, trace, exp, sum, mean, var, abs, sqrt
from numpy import inf, finfo, PINF
from scipy import linalg

from constant import *
from util import history


def firingrate(h, m, v, a, b):
    eta_x = einsum('ijk, ki->ji', h, b) + m.dot(a) + 0.5 * v.dot(a ** 2)
    np.clip(eta_x, MIN_EXP, MAX_EXP, out=eta_x)
    return np.exp(eta_x)


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
        GTWG = G.T.dot(w[:, l].reshape((T, 1)) * G)
        v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True)))),
                      axis=1)
    return v


def elbo(y, h, chol, m, w, v, a, b):
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
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))

    lb = sum(y * eta - lam)

    for l in range(L):
        G = chol[l, :]
        GTWG = G.T.dot(w[:, l].reshape((T, 1)) * G)
        A = GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True))
        m_div_G = linalg.lstsq(G, m[:, l])[0]
        tr = T - trace(GTWG) + trace(A)
        lndet = np.linalg.slogdet(eyer - GTWG + A)[1]

        lb += -0.5 * inner(m_div_G, m_div_G) - 0.5 * tr + 0.5 * lndet

    return lb


def accumulate(accu, grad, decay):
    """adagrad
    Args:
        accu: accumulation matrix
        grad: new gradient
        decay: expoential decay

    Returns:

    """
    return decay * accu + (1 - decay) * grad ** 2


def train(y, lag, chol, m0=None, a0=None, b0=None, niter=50, tol=1e-5, decay=0.95, eps=1e-6, verbose=True):
    """Variational Bayesian
    Args:
        y: observations (T, N), spikes
        lag: order of autocorrelation
        chol: cholesky factorizations of prior covariances (L, T, r)
        m0: initial posterior mean (T, L)
        a0: initial latent coefficients (L, N)
        b0: initial autocorrelation coefficients (1 + lag, N)
        niter: max number of iterations
        tol: relative tolerance of convergence
        decay: adagrad
        eps: adagrad
        verbose: detailed output

    Returns:
        lbound: ELBO of all iterations
        m: posterior mean (T, L)
        lv: L matrices in factorization of V = LL' (L, T, r)
        a: latent coefficients (L, N)
        b: autocorrelation coefficients (1 + lag, N)
        elapsed: running time
        converged: whether the algorithm converged within iteration limit
    """
    L, T, r = chol.shape
    N = y.shape[1]
    eyek = identity(r)

    h = history(y, lag)

    if m0 is None:
        m0 = tile(mean(y, axis=1), (1, L))

    if a0 is None:
        a0 = linalg.lstsq(m0, y)[0]

    if b0 is None:
        b0 = empty((1 + lag, N), dtype=float)
        for n in range(N):
            b0[:, n] = linalg.lstsq(h[n, :], y[:, n])[0]

    a = a0
    b = b0
    m = m0

    R = empty_like(y, dtype=float)

    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp(eta.clip(MIN_EXP, MAX_EXP))  # no a'Va at beginning
    w = lam.dot(a.T ** 2)
    v = vfromw(w, chol)

    # backup, could be useless
    good_m = m.copy()
    good_a = a.copy()
    good_b = b.copy()
    #

    lbound = full(niter, fill_value=finfo(float).min, dtype=float)
    lbound[0] = elbo(y, h, chol, m, w, v, a, b)

    # adagrad
    accu_grad_a = zeros_like(a)
    accu_grad_b = zeros_like(b)
    accu_grad_m = zeros_like(m)

    # dec = False
    converged = False
    i = 1
    start = timeit.default_timer()
    while not converged and i < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(i))

        # estimate latent
        for l in range(L):
            # m
            eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
            lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
            G = chol[l]
            grad_m = (y - lam).dot(a[l, :]) - linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]
            accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m, decay)
            # accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m / linalg.norm(grad_m, ord=inf), decay)

            wada = (w[:, l] + sqrt(eps + accu_grad_m[:, l])).reshape((T, 1))  # adjusted by adagrad
            GTWG = G.T.dot(wada * G)

            u = G.dot(G.T.dot((y - lam).dot(a[l, :]))) - m[:, l]
            delta_m = u - G.dot((wada * G).T.dot(u)) + \
                      G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (wada * G).T.dot(u), sym_pos=True)))

            m[:, l] = good_m[:, l] + delta_m
            m[:, l] -= mean(m[:, l])
            scale = linalg.norm(m[:, l], ord=inf)
            a[l, :] *= scale
            m[:, l] /= scale

        # estimate coefficients
        for n in range(N):
            eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
            lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
            # a
            va = v * a[:, n]  # (T, L)
            wv = diag(lam[:, n].dot(v))
            grad_a = m.T.dot(y[:, n]) - (m + va).T.dot(lam[:, n])
            accu_grad_a[:, n] = accumulate(accu_grad_a[:, n], grad_a, decay)

            neghess_a = (m + va).T.dot(lam[:, n, newaxis] * (m + va)) + wv
            delta_a = linalg.solve(neghess_a, grad_a, sym_pos=True)
            a[:, n] += delta_a

            # b
            grad_b = h[n, :].T.dot(y[:, n] - lam[:, n])
            accu_grad_b[:, n] = accumulate(accu_grad_b[:, n], grad_b, decay)
            neghess_b = h[n, :].T.dot(lam[:, n, newaxis] * h[n, :])
            b[:, n] += linalg.solve(neghess_b, grad_b, sym_pos=True)

        # update w
        eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
        vhat = var(y - eta, axis=0, ddof=0)
        lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
        w = lam.dot(a.T ** 2)
        v = vfromw(w, chol)

        lbound[i] = elbo(y, h, chol, m, w, v, a, b)

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'
                  'change in a = {:.10f}\n'
                  'change in b = {:.10f}\n'
                  'change in m = {:.10f}\n'.format(lbound[i], lbound[i] - lbound[i - 1],
                                                   timeit.default_timer() - iter_start,
                                                   linalg.norm(good_a - a, ord=inf),
                                                   linalg.norm(good_b - b, ord=inf),
                                                   linalg.norm(good_m - m, ord=inf)))

        # check convergence
        if abs(lbound[i] - lbound[i - 1]) < tol * abs(lbound[i - 1]):
            converged = True

        # keep a copy of current iteration
        # no use in algorithm
        # for debug only now
        good_m[:] = m
        good_a[:] = a
        good_b[:] = b

        i += 1

    stop = timeit.default_timer()

    lv = empty((L, T, r), dtype=float)  # V = LL'
    for l in range(L):
        G = chol[l, :]
        GTWG = G.T.dot(w[:, l].reshape((T, 1)) * G)

        A = eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))  # A should be pd but numerically not
        eigval, eigvec = linalg.eigh(A)
        eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
        lv[l, :] = G.dot(eigvec.dot(diag(sqrt(eigval))))

    return lbound[:i], m, lv, a, b, stop - start, converged
