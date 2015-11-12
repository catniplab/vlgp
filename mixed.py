import timeit

import numpy as np
from numpy import empty, empty_like, full, zeros, zeros_like, newaxis, tile
from numpy import identity, diag, einsum, inner, trace, exp, sum, mean, var, abs, sqrt
from numpy import inf, finfo, PINF
from scipy import linalg

from util import history

# lower and upper bound of exp
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
        GTWG = G.T.dot(w[:, l].reshape((T, 1)) * G)
        v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True)))),
                      axis=1)
    return v


def elbo(y, h, family, chol, m, w, v, a, b, vhat):
    """Evidence Lower BOund

    Args:
        y: observations (T, N)
        h: autocorrelation regressor (N, T, 1 + p)
        family: distributions (N)
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
    poisson = family == 'poisson'
    gaussian = family == 'gaussian'

    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))

    lpois = sum(y[:, poisson] * eta[:, poisson] - lam[:, poisson])

    lgauss = - 0.5 * sum(((y[:, gaussian] - eta[:, gaussian]) ** 2 +
                         v.dot(a[:, gaussian] ** 2)) / vhat[gaussian])

    lb = lpois + lgauss

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
    # return decay * accu + (1 - decay) * grad ** 2
    return accu + grad ** 2


def train(y, family, lag, chol, m0=None, a0=None, b0=None, abest=True,
          niter=50, tol=1e-5, adagrad=15, decay=0.95, eps=1e-6, verbose=True):
    """Variational Bayesian

    Args:
        y: observations (T, N), spikes or continuous
        family: distributions (N), 'poisson' and 'gaussian' supported now
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

    poisson = family == 'poisson'
    gaussian = family == 'gaussian'

    h = history(y, lag)

    if m0 is None:
        m0 = tile(mean(y, axis=1), (1, L))
        m0 -= mean(m0, axis=0)

    if a0 is None:
        a0 = linalg.lstsq(m0, y)[0]

    if b0 is None:
        b0 = empty((1 + lag, N), dtype=float)
        for n in range(N):
            b0[:, n] = linalg.lstsq(h[n, :], y[:, n])[0]

    a = a0.copy()
    b = b0.copy()
    m = m0.copy()

    if not abest:
        a.setflags(write=0)
        b.setflags(write=0)

    U = empty((T, N), dtype=float)
    R = empty_like(y, dtype=float)

    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    vhat = var(y - eta, axis=0, ddof=0)
    lam = exp(eta.clip(LB, UB))  # no a'Va at beginning
    U[:, poisson] = lam[:, poisson]
    U[:, gaussian] = 1 / vhat[gaussian]
    w = U.dot(a.T ** 2)
    v = vfromw(w, chol)

    # backup, could be useless
    good_m = m.copy()
    good_a = a.copy()
    good_b = b.copy()
    good_w = w.copy()
    #

    lbound = full(niter, fill_value=finfo(float).min, dtype=float)
    lbound[0] = elbo(y, h, family, chol, m, w, v, a, b, vhat)

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
            eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
            vhat = var(y - eta, axis=0, ddof=0)
            lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
            G = chol[l]
            grad_m = (y[:, poisson] - lam[:, poisson]).dot(a[l, poisson]) + \
                     ((y[:, gaussian] - eta[:, gaussian]) /
                      vhat[gaussian]).dot(a[l, gaussian]) - linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]
            accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m, decay)
            # accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m / linalg.norm(grad_m, ord=inf), decay)

            if i > adagrad:
                wada = (w[:, l] + sqrt(eps + accu_grad_m[:, l])).reshape((T, 1))  # adjusted by adagrad
            else:
                wada = w[:, l].reshape((T, 1))
            GTWG = G.T.dot(wada * G)

            # eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
            # lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(EXP_LB, EXP_UB))
            R[:, poisson] = y[:, poisson] - lam[:, poisson]
            R[:, gaussian] = (y[:, gaussian] - eta[:, gaussian]) / vhat[gaussian]

            u = G.dot(G.T.dot(R.dot(a[l, :]))) - m[:, l]
            delta_m = u - G.dot((wada * G).T.dot(u)) + \
                      G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (wada * G).T.dot(u), sym_pos=True)))

            m[:, l] = good_m[:, l] + delta_m
            m[:, l] -= mean(m[:, l])
            scale = linalg.norm(m[:, l], ord=inf)
            m[:, l] /= scale
            if abest:
                a[l, :] *= scale

        # estimate coefficients
        if abest:
            for n in range(N):
                eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
                lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
                if family[n] == 'poisson':
                    # a
                    va = v * a[:, n]  # (T, L)
                    wv = diag(lam[:, n].dot(v))
                    grad_a = m.T.dot(y[:, n]) - (m + va).T.dot(lam[:, n])
                    accu_grad_a[:, n] = accumulate(accu_grad_a[:, n], grad_a, decay)

                    neghess_a = (m + va).T.dot(lam[:, n, newaxis] * (m + va)) + wv
                    if i > adagrad:
                        delta_a = linalg.solve(neghess_a + diag(sqrt(eps + accu_grad_a[:, n])), grad_a, sym_pos=True)
                    else:
                        delta_a = linalg.solve(neghess_a, grad_a, sym_pos=True)
                    a[:, n] += delta_a

                    # b
                    grad_b = h[n, :].T.dot(y[:, n] - lam[:, n])
                    accu_grad_b[:, n] = accumulate(accu_grad_b[:, n], grad_b, decay)
                    neghess_b = h[n, :].T.dot(lam[:, n, newaxis] * h[n, :])
                    if i > adagrad:
                        b[:, n] += linalg.solve(neghess_b + diag(sqrt(eps + accu_grad_b[:, n])), grad_b, sym_pos=True)
                    else:
                        b[:, n] += linalg.solve(neghess_b, grad_b, sym_pos=True)
                else:
                    # a's least squares solution for Gaussian channel
                    # (m'm + diag(j'v))^-1 m'(y - Hb)
                    a[:, n] = linalg.solve(m.T.dot(m) + diag(sum(v, axis=0)), m.T.dot(y[:, n] - h[n, :].dot(b[:, n])),
                                           sym_pos=True)

                    # b's least squares solution for Gaussian channel
                    # (H'H)^-1 H'(y - ma)
                    b[:, n] = linalg.solve(h[n, :].T.dot(h[n, :]), h[n, :].T.dot(y[:, n] - m.dot(a[:, n])), sym_pos=True)

        # update w
        eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
        vhat = var(y - eta, axis=0, ddof=0)
        lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
        U[:, poisson] = lam[:, poisson]
        U[:, gaussian] = 1 / vhat[gaussian]
        w = U.dot(a.T ** 2)
        v = vfromw(w, chol)

        lbound[i] = elbo(y, h, family, chol, m, w, v, a, b, vhat)

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

        if i > adagrad and lbound[i] < lbound[i - 1]:
            print('adagrad enabled')
            m[:] = good_m
            a[:] = good_a
            b[:] = good_b
            w[:] = good_w

        # keep a copy of current iteration
        # no use in algorithm
        # for debug only now
        good_m[:] = m
        good_a[:] = a
        good_b[:] = b
        good_w[:] = w

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

    if verbose:
        print('Exit')

    return lbound[:i], m, lv, a, b, stop - start, converged
