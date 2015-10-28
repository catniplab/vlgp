import timeit
import numpy as np
from scipy import linalg
from util import selfhistory


def elbo(y, h, dist, chol, m, v, a, b):
    L, T, k = chol.shape
    N = y.shape[1]
    eyek = np.identity(k)

    eta = np.einsum('ijk, ik->ij', h, b).T + m.dot(a)
    lam = np.exp((eta + 0.5 * v.dot(a ** 2)).clip(-30, 30))
    var = np.var(y - eta, axis=0, ddof=0)

    lpois = np.sum(y[:, dist == 'poisson'] * eta[:, dist == 'poisson'] - lam[:, dist == 'poisson'])

    lgauss = - 0.5 * np.sum((y[:, dist == 'gaussian'] - eta[:, dist == 'gaussian']) ** 2 / var[dist == 'gaussian'] +
                            v.dot(a[:, dist == 'gaussian'] ** 2))

    lb = lpois + lgauss

    for l in range(L):
        G = chol[l, :]
        u = np.empty((T, N), dtype=float)
        u[:, dist == 'poisson'] = lam[:, dist == 'poisson']
        u[:, dist == 'gaussian'] = 1 / var[dist == 'gaussian']
        w = u.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        m_div_G = linalg.lstsq(G, m[:, l])[0]
        trace = (T - np.trace(GTWG) + np.trace(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))
        lndet = np.linalg.slogdet(eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]

        lb += -0.5 * np.inner(m_div_G, m_div_G) - 0.5 * trace + 0.5 * lndet

    return lb


def accumulate(accu, grad, decay):
    return decay * accu + (1 - decay) * grad ** 2


def residual(y, dist, h, m, v, a, b):
    res = np.empty_like(y, dtype=float)
    eta = np.einsum('ijk, ik->ij', h, b).T + m.dot(a)
    lam = np.exp((eta + 0.5 * v.dot(a ** 2)).clip(-30, 30))
    var = np.var(y - eta, axis=0, ddof=0)
    res[:, dist == 'poisson'] = y[:, dist == 'poisson'] - lam[:, dist == 'poisson']
    res[:, dist == 'gaussian'] = (y[:, dist == 'gaussian'] - eta[:, dist == 'gaussian']) / var[dist == 'gaussian']
    return res


def train(y, dist, p, chol, m0=None, a0=None, b0=None, niter=50, chkcnv=5, tol=1e-4, decay=0.9, eps=1e-6, verbose=True):
    def updatev():
        eta = np.einsum('ijk, ik->ij', h, b).T + m.dot(a)
        lam = np.exp((eta + 0.5 * v.dot(a ** 2)).clip(-30, 30))
        var = np.var(y - eta, axis=0, ddof=0)
        for l in range(L):
            G = chol[l, :]
            u = np.empty((T, N), dtype=float)
            u[:, dist == 'poisson'] = lam[:, dist == 'poisson']
            u[:, dist == 'gaussian'] = 1 / var[dist == 'gaussian']
            w = u.dot(a[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)

    L, T, cholrank = chol.shape
    N = y.shape[1]
    eyek = np.identity(cholrank)

    y0 = np.zeros(N, dtype=float)
    y0[dist == 'gaussian'] = np.mean(y[:, dist == 'gaussian'], axis=0)
    h = selfhistory(y, p, y0)

    if m0 is None:
        m0 = np.tile(np.mean(y, axis=1), (1, L))

    if a0 is None:
        a0 = linalg.lstsq(m0, y)[0]

    if b0 is None:
        b0 = np.empty((N, 1 + p), dtype=float)
        for n in range(N):
            b0[n, :] = linalg.lstsq(h[n, :], y[:, n])[0]

    a = a0
    b = b0
    m = m0

    good_m = m.copy()
    good_a = a.copy()
    good_b = b.copy()

    v = np.empty_like(m, dtype=float)

    lbound = np.full(niter, fill_value=np.finfo(float).min, dtype=float)
    lbound[0] = elbo(y, h, dist, chol, m, v, a, b)

    # adagrad
    accu_grad_a = np.zeros_like(a)
    accu_grad_b = np.zeros_like(b)
    accu_grad_m = np.zeros_like(m)

    # dec = False
    converged = False
    i = 1
    start = timeit.default_timer()
    while not converged and i < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(i))

        # estimate latent
        eta = np.einsum('ijk, ik->ij', h, b).T + m.dot(a)
        lam = np.exp((eta + 0.5 * v.dot(a ** 2)).clip(-30, 30))
        var = np.var(y - eta, axis=0, ddof=0)
        for l in range(L):
            # m
            G = chol[l]
            grad_m = (y[:, dist == 'poisson'] - lam[:, dist == 'poisson']).dot(a[l, dist == 'poisson']) + \
                     ((y[:, dist == 'gaussian'] - eta[:, dist == 'gaussian']) / var[dist == 'gaussian']).dot(a[l, dist == 'gaussian']) - \
                     linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]
            accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m, decay)

            u = np.empty((T, N), dtype=float)
            u[:, dist == 'poisson'] = lam[:, dist == 'poisson']
            u[:, dist == 'gaussian'] = 1 / var[dist == 'gaussian']
            w = (u.dot(a[l, :] ** 2) + np.sqrt(eps + accu_grad_m[:, l])).reshape((T, 1))
            GTWG = G.T.dot(w * G)

            # adj4grad = np.ones(N, dtype=float)
            # adj4grad[dist == 'gaussian'] = 1 / var[dist == 'gaussian']
            # residual = adj4grad * (y - eta)
            R = residual(y, dist, h, m, v, a, b)

            u = G.dot(G.T.dot(R.dot(a[l, :]))) - m[:, l]
            delta_m = u - G.dot((w * G).T.dot(u)) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (w * G).T.dot(u),
                                                                                  sym_pos=True)))
            m[:, l] = good_m[:, l] + delta_m
            m[:, l] -= np.mean(m[:, l])
            m[:, l] /= linalg.norm(m[:, l], ord=np.inf)

            # grad_a = np.empty(N, dtype=float)
            # neg_diag_Ha = np.empty_like(grad_a)
            # grad_a[dist] = (y[:, dist] - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * a[l, dist]
            # grad_a[dist == 'gaussian'] = ((y[:, dist == 'gaussian'] - eta[:, dist == 'gaussian']).T.dot(m[:, l]) - np.sum(v[:, l]) * a[l, dist == 'gaussian']) / vgauss[dist == 'gaussian']
            # accu_grad_a[l, :] = decay * accu_grad_a[l, :] + (1 - decay) * grad_a ** 2
            # neg_diag_Ha[dist] = np.sum(lam * ((m[:, l, np.newaxis] + np.outer(v[:, l], a[l, dist])) ** 2), axis=0) + \
            #               lam.T.dot(v[:, l])
            # neg_diag_Ha[dist == 'gaussian'] = np.sum(m[:, l] ** 2 + v[:, l]) / vgauss[dist == 'gaussian']
            # delta_a = grad_a / (np.sqrt(eps + accu_grad_a[l, :]) + neg_diag_Ha)
            # a[l, :] += delta_a
        updatev()

        # estimate coefficients
        eta = np.einsum('ijk, ik->ij', h, b).T + m.dot(a)
        lam = np.exp((eta + 0.5 * v.dot(a ** 2)).clip(-30, 30))
        var = np.var(y - eta, axis=0, ddof=0)
        for n in range(N):
            neghess_b = np.zeros((1 + p, 1 + p), dtype=float)
            if dist[n] == 'poisson':
                grad_b = (y[:, n] - lam[:, n]).dot(h[n, :])
                for t in range(T):
                    neghess_b += lam[t, n] * np.outer(h[n, t, :], h[n, t, :])
            else:
                grad_b = ((y[:, n] - eta[:, n]) / var[n]).dot(h[n, :])
                for t in range(T):
                    neghess_b += np.outer(h[n, t, :], h[n, t, :]) / var[n]
            accu_grad_b[n, :] = accumulate(accu_grad_b[n, :], grad_b, decay)

            delta_b = linalg.solve(neghess_b + np.diag(np.sqrt(eps + accu_grad_b[n, :])), grad_b, sym_pos=True)
            b[n, :] += delta_b

            neghess_a = np.zeros((L, L), dtype=float)
            if dist[n] == 'poisson':
                grad_a = (y[:, n] - lam[:, n]).dot(m) - lam[:, n].dot(v * a[:, n])
                for t in range(T):
                    neghess_a += lam[t, n] * (np.outer(m[t, :] + v[t, :] * a[:, n], m[t, :] + v[t, :] * a[:, n]) +
                                              np.diag(v[t, :]))
            else:
                grad_a = ((y[:, n] - eta[:, n]).dot(m) - np.sum(v * a[:, n], axis=0)) / var[n]
                for t in range(T):
                    neghess_a += (np.outer(m[t, :], m[t, :]) + np.diag(v[t, :])) / var[n]
            accu_grad_a[:, n] = accumulate(accu_grad_a[:, n], grad_a, decay)

            delta_a = linalg.solve(neghess_a + np.diag(np.sqrt(eps + accu_grad_a[:, n])), grad_a, sym_pos=True)
            a[:, n] += delta_a
        updatev()

        lbound[i] = elbo(y, h, dist, chol, m, v, a, b)

        # check convergence
        change_a = np.max(np.abs(good_a - a))
        change_b = np.max(np.abs(good_b - b))
        change_m = np.max(np.abs(good_m - m))

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'
                  'change in a = {:.10f}\n'
                  'change in b = {:.10f}\n'
                  'change in m = {:.10f}\n'.format(lbound[i], lbound[i] - lbound[i - 1],
                                                   timeit.default_timer() - iter_start,
                                                   change_a, change_b, change_m))

        if i > chkcnv and np.abs(lbound[i] - lbound[i - 1]) < tol * np.abs(lbound[i - 1]):
            converged = True

        good_m[:] = m
        good_a[:] = a
        good_b[:] = b

        i += 1

    stop = timeit.default_timer()

    lv = np.empty((L, T, cholrank), dtype=float)
    eta = np.einsum('ijk, ik->ij', h, b).T + m.dot(a)
    lam = np.exp((eta + 0.5 * v.dot(a ** 2)).clip(-30, 30))

    for l in range(L):
        G = chol[l, :]
        u = np.empty((T, N), dtype=float)
        u[:, dist == 'poisson'] = lam[:, dist == 'poisson']
        u[:, dist == 'gaussian'] = 1 / var[dist == 'gaussian']
        w = u.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)

        A = eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))  # A should be pd but numerically not
        eigval, eigvec = linalg.eigh(A)
        eigval.clip(0, np.PINF, out=eigval)  # remove negative eigenvalues
        lv[l, :] = G.dot(eigvec.dot(np.diag(np.sqrt(eigval))))

    return lbound[:i], m, lv, a, b, stop - start, converged
