import timeit
import numpy as np
from scipy import linalg
from util import makeregressor


def elbo(y, h, chol, m, v, a, b, K):
    T, N = y.shape
    L, _, k = chol.shape
    eyek = np.identity(k)

    out = - 0.5 * np.linalg.slogdet(K)[1] * T
    for t in range(T):
        r = y[t, :] - m[t, :].dot(a) - h[t, :].dot(b)
        out += -0.5 * r.dot(linalg.solve(K, r, sym_pos=True)) \
              - 0.5 * linalg.solve(K, a.T.dot(np.diag(v[t, :])).dot(a), sym_pos=True).trace()

    for l in range(L):
        G = chol[l, :]
        w = np.full(T, fill_value=a[l, :].T.dot(linalg.solve(K, a[l, :])), dtype=float)
        GTWG = G.T.dot(w[..., np.newaxis] * G)
        m_div_G = linalg.lstsq(G, m[:, l])[0]
        out += - 0.5 * np.inner(m_div_G, m_div_G) \
               - 0.5 * (T - np.trace(GTWG) + np.trace(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))) \
               + 0.5 * np.linalg.slogdet(eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]
    return out


def train(y, p, chol, m0=None, a0=None, b0=None, bmask=None, K0=None, niter=50, tol=1e-5, verbose=True):
    T, N = y.shape
    L, _, k = chol.shape
    eyek = np.identity(k)

    h = makeregressor(y, p)

    if m0 is None:
        m0 = np.zeros((T, L), dtype=float)
    m = m0.copy()

    if a0 is None:
        a0 = np.zeros((N, L), dtype=float)
    a = a0.copy()

    if b0 is None:
        b0 = linalg.lstsq(h, y)[0]
    b = b0.copy()

    if bmask is None:
        bmask = np.full_like(b, fill_value=True, dtype=bool)

    if K0 is None:
        K0 = np.eye(N, dtype=float)
    K = K0.copy()

    v = np.ones((T, L)) * chol[:, 0, 0] ** 2

    chosen = np.full(L, fill_value=True, dtype=bool)  #

    #
    lb = np.full(niter, fill_value=np.finfo(float).min, dtype=float)
    lb[0] = elbo(y, h, chol, m, v, a, b, K)

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
            z = (y - h.dot(b) - m[:, chosen].dot(a[chosen, :])).dot(linalg.solve(K, a[l, :]))
            G = chol[l, :]
            w = np.full(T, fill_value=a[l, :].T.dot(linalg.solve(K, a[l, :])), dtype=float)
            GTWG = G.T.dot(w[..., np.newaxis] * G)
            m[:, l] = (G - G.dot(GTWG) + G.dot(GTWG).dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))).dot(G.T.dot(z))
            m[:, l] -= np.mean(m[:, l])
            m[:, l] /= linalg.norm(m[:, l], ord=np.inf)
            v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)

        hra = np.zeros((b.shape[0], N), dtype=float)
        hh = np.zeros((b.shape[0], b.shape[0]), dtype=float)
        for t in range(T):
            hra += np.outer(h[t, :], y[t, :] - m[t, :].dot(a))
            hh += np.outer(h[t, :], h[t, :])
        b = linalg.solve(hh, hra, sym_pos=True)

        mrb = np.zeros((L, N), dtype=float)
        mmv = np.zeros((L, L), dtype=float)
        for t in range(T):
            mrb = np.outer(m[t, :], y[t, :] - h[t, :].dot(b))
            mmv = np.outer(m[t, :], m[t, :]) + np.diag(v[t, :])
        a = linalg.solve(mmv, mrb, sym_pos=True)

        K = np.zeros((N, N), dtype=float)
        for t in range(T):
            r = y[t, :] - m[t, :].dot(a) - h[t, :].dot(b)
            K += np.outer(r, r) + a.T.dot(np.diag(v[t, :])).dot(a)
        K /= T
        lb[i] = elbo(y, h, chol, m, v, a, b, K)

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'.format(lb[i], lb[i] - lb[i - 1], timeit.default_timer() - iter_start))

        if np.abs(lb[i] - lb[i - 1]) < tol * np.abs(lb[i - 1]):
            converged = True

        i += 1
    stop = timeit.default_timer()

    return elbo, m, v, a, b, K, stop - start, converged
