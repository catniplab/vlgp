import timeit
import numpy as np
from scipy import linalg
from util import selfhistory


def elbo(y, h, pois, chol, m, v, a, b, vgauss):
    L, T, k = chol.shape
    N = y.shape[1]
    eyek = np.identity(k)

    eta = np.einsum('ijk, ik->ij', h, b) + m.dot(a)
    lam = np.exp((eta[pois] + 0.5 * v.dot(a[:, pois] ** 2)).clip(-30, 30))
    lpois = np.sum(y[:, pois] * eta[:, pois] - lam)

    lgauss = - 0.5 * np.sum((y[:, ~pois] - eta[:, ~pois]) ** 2 / vgauss[~pois] + v.dot(a[:, ~pois] ** 2))

    lb = lpois + lgauss

    for l in range(L):
        G = chol[l, :]
        adj = np.empty((T, N), dtype=float)
        adj[:, pois] = lam
        adj[:, ~pois] = 1 / vgauss[~pois]
        w = adj.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        m_div_G = linalg.lstsq(G, m[:, l])[0]
        trace = (T - np.trace(GTWG) + np.trace(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))
        lndet = np.linalg.slogdet(eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]

        lb += -0.5 * np.inner(m_div_G, m_div_G) - 0.5 * trace + 0.5 * lndet

    return lb


def train(y, pois, p, chol, m0=None, a0=None, b0=None, niter=50, tol=1e-5,
          verbose=True):
    def updatev():
        eta = np.einsum('ijk, ik->ij', h, b) + m.dot(a)
        lam = np.exp((eta[pois] + 0.5 * v.dot(a[:, pois] ** 2)).clip(-30, 30))
        for l in range(L):
            G = chol[l, :]
            adj = np.empty((T, N), dtype=float)
            adj[:, pois] = lam
            adj[:, ~pois] = 1 / vgauss[~pois]
            w = adj.dot(a[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)

    L, T, cholrank = chol.shape
    N = y.shape[1]
    eyek = np.identity(cholrank)
    h = selfhistory(y, p)

    if a0 is None:
        a0 = np.zeros((L, N), dtype=float)
    if b0 is None:
        b0 = linalg.lstsq(h, y)[0]
    if m0 is None:
        m0 = np.zeros((L, T), dtype=float)

    a = a0
    b = b0
    m = m0

    vgauss = np.var((y - h.dot(b) - m.dot(a)), axis=1)

    good_m = m.copy()
    good_a = a.copy()
    good_b = b.copy()

    v = np.empty_like(m, dtype=float)

    lb = np.full(niter, fill_value=np.finfo(float).min, dtype=float)
    lb[0] = 0

    # adagrad
    decay = 0.9
    eps = 1e-6
    accu_grad_a = np.zeros_like(a)

    converged = False
    i = 1
    start = timeit.default_timer()
    while not converged and i < niter:
        # estimate b
        eta = np.einsum('ijk, ik->ij', h, b) + m.dot(a)
        for n in range(N):
            neghess = np.zeros((1 + p, 1 + p), dtype=float)
            if pois[n]:
                lam = np.exp((eta[:, n] + 0.5 * v.dot(a[:, n] ** 2)).clip(-30, 30))
                grad = (y[:, n] - lam).dot(h[n, :])
                for t in range(T):
                    neghess += lam[t] * np.outer(h[n, t, :], h[n, t, :])
            else:
                grad = ((y[:, n] - eta[:, n]) / vgauss[n]).dot(h[n, :])
                for t in range(T):
                    neghess += np.outer(h[n, t, :], h[n, t, :])
                neghess /= vgauss[n]
            delta = linalg.lstsq(neghess, grad)[0]
            b[n] += delta
        updatev()

        # estimate latent
        eta = np.einsum('ijk, ik->ij', h, b) + m.dot(a)
        lam = np.exp((eta[pois] + 0.5 * v.dot(a[:, pois] ** 2)).clip(-30, 30))
        for l in range(L):
            G = chol[l]
            adj = np.empty((T, N), dtype=float)
            adj[:, pois] = lam
            adj[:, ~pois] = 1 / vgauss[~pois]
            w = lam.dot(a[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)

            adj4grad = np.ones(N, dtype=float)
            adj4grad[~pois] = 1 / vgauss[~pois]
            residual = adj4grad * (y - eta)

            u = G.dot(G.T.dot(residual.dot(a[l, :]))) - m[:, l]
            delta_m = u - G.dot((w * G).T.dot(u)) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (w * G).T.dot(u),
                                                                                  sym_pos=True)))
            m[:, l] = good_m[:, l] + delta_m
            m[:, l] -= np.mean(m[:, l])
            m[:, l] /= linalg.norm(m[:, l], ord=np.inf)
            lb = elbo(y, h, pois, chol, m, v, a, b, vgauss)
            if np.isfinite(lb) and lb > good_elbo:
                # Newton step
                good_elbo = lb
            else:
                m[:, l] = good_m[:, l]

            grad_a = np.empty(N, dtype=float)
            neg_diag_Ha = np.empty_like(grad_a)
            grad_a[pois] = (y[:, pois] - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * a[l, pois]
            grad_a[~pois] = ((y[:, ~pois] - eta[:, ~pois]).T.dot(m[:, l]) - np.sum(v[:, l]) * a[l, ~pois]) / vgauss[~pois]
            accu_grad_a[l, :] = decay * accu_grad_a[l, :] + (1 - decay) * grad_a ** 2
            neg_diag_Ha[pois] = np.sum(lam * ((m[:, l, np.newaxis] + np.outer(v[:, l], a[l, :])) ** 2), axis=0) + \
                          lam.T.dot(v[:, l])
            neg_diag_Ha[~pois] = np.sum(m[:, l] ** 2 + v[:, l]) / vgauss[~pois]
            delta_a = grad_a / (np.sqrt(eps + accu_grad_a[l, :]) + neg_diag_Ha)
            a[l, :] += delta_a
        updatev()

        # estimate gaussian variance
        eta = np.einsum('ijk, ik->ij', h, b) + m.dot(a)
        vgauss = np.mean((y - eta) ** 2, axis=1)

        lb[i] = elbo(y, h, pois, chol, m, v, a, b, vgauss)

        if np.abs(lb[i] - lb[i - 1]) < tol * np.abs(lb[i - 1]):
            converged = True

        good_m[:] = m
        good_a[:] = a
        good_b[:] = b

        i += 1

    stop = timeit.default_timer()
    return lb, m, v, a, b, stop - start, converged
