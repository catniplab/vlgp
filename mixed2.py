import timeit
import numpy as np
from scipy import linalg
from util import selfhistory
from numpy import identity, diag, eye, dot, einsum, inner, outer, trace, exp, log, sum, mean, var, min, max, abs, sqrt
from numpy import empty, empty_like, full, full_like, zeros, zeros_like, ones, ones_like, newaxis, tile
from numpy import inf, finfo, PINF, NINF

LB = -20
UB = 20


def vfromw(w, chol):
    L, T, k = chol.shape
    eyek = identity(k)
    v = empty((T, L), dtype=float)
    for l in range(L):
        G = chol[l, :]
        GTWG = G.T.dot(w[:, l].reshape((T, 1)) * G)
        v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                         axis=1)
    return v


def elbo(y, h, family, chol, m, w, v, a, b, vhat):
    L, T, k = chol.shape
    eyek = identity(k)
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
        A = GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))
        m_div_G = linalg.lstsq(G, m[:, l])[0]
        tr = T - trace(GTWG) + trace(A)
        lndet = np.linalg.slogdet(eyek - GTWG + A)[1]

        lb += -0.5 * inner(m_div_G, m_div_G) - 0.5 * tr + 0.5 * lndet

    return lb


def accumulate(accu, grad, decay):
    return decay * accu + (1 - decay) * grad ** 2


def residual(y, h, family, chol, m, w, v, a, b, vhat):
    poisson = family == 'poisson'
    gaussian = family == 'gaussian'
    res = empty_like(y, dtype=float)
    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
    res[:, poisson] = y[:, poisson] - lam[:, poisson]
    res[:, gaussian] = (y[:, gaussian] - eta[:, gaussian]) / vhat[gaussian]
    return res


def train(y, family, p, chol, m0=None, a0=None, b0=None, niter=50, tol=1e-4, decay=0.9, eps=1e-6, verbose=True):
    L, T, cholrank = chol.shape
    N = y.shape[1]
    eyek = identity(cholrank)

    poisson = family == 'poisson'
    gaussian = family == 'gaussian'

    y0 = zeros(N, dtype=float)
    y0[gaussian] = mean(y[:, gaussian], axis=0)
    h = selfhistory(y, p, y0)

    if m0 is None:
        m0 = tile(mean(y, axis=1), (1, L))

    if a0 is None:
        a0 = linalg.lstsq(m0, y)[0]

    if b0 is None:
        b0 = empty((1 + p, N), dtype=float)
        for n in range(N):
            b0[:, n] = linalg.lstsq(h[n, :], y[:, n])[0]

    a = a0
    b = b0
    m = m0
    a2 = a.copy()
    m2 = m.copy()

    good_m = m.copy()
    good_a = a.copy()
    good_b = b.copy()

    w = zeros((T, L), dtype=float)
    v = vfromw(w, chol)
    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
    vhat = var(y - eta, axis=0, ddof=0)
    for l in range(L):
        U = empty((T, N), dtype=float)
        U[:, poisson] = lam[:, poisson]
        U[:, gaussian] = 1 / vhat[gaussian]
        w[:, l] = U.dot(a[l, :] ** 2)
    v = vfromw(w, chol)

    vhat = var(y - einsum('ijk, ki->ji', h, b) + m.dot(a), axis=0, ddof=0)
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
        m2[:] = good_m
        a2[:] = good_a
        for l in range(L):
            eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
            lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
            vhat = var(y - eta, axis=0, ddof=0)
            # m
            G = chol[l]
            grad_m = (y[:, poisson] - lam[:, poisson]).dot(a[l, poisson]) + \
                     ((y[:, gaussian] - eta[:, gaussian]) / vhat[gaussian]).dot(a[l, gaussian]) \
                     - linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]
            # accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m, decay)
            accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m / linalg.norm(grad_m, ord=inf), decay)

            w1 = (w[:, l] + sqrt(eps + accu_grad_m[:, l])).reshape((T, 1))  # adjusted by adagrad
            GTWG = G.T.dot(w1 * G)

            R = residual(y, h, family, chol, m, w, v, a, b, vhat)

            u = G.dot(G.T.dot(R.dot(a[l, :]))) - m[:, l]
            delta_m = u - G.dot((w1 * G).T.dot(u)) + \
                      G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (w1 * G).T.dot(u), sym_pos=True)))

            # lb1 = elbo(y, h, family, chol, m2, w, v, a, b, vhat)
            # m2[:, l] += 1e-10 * grad_m / linalg.norm(grad_m, ord=inf)
            # lb2 = elbo(y, h, family, chol, m2, w, v, a, b, vhat)
            # if lb2 < lb1:
            #     print('Inc m[{}] tiny grad = {}'.format(l, lb2 - lb1))

            m[:, l] = good_m[:, l] + delta_m
            m[:, l] -= mean(m[:, l])
            scale = linalg.norm(m[:, l], ord=inf)
            a[l, :] *= scale
            m[:, l] /= scale

            # grad_a = empty(N, dtype=float)
            # neg_diag_Ha = empty_like(grad_a)
            # grad_a[family] = (y[:, family] - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * a[l, family]
            # grad_a[gaussian] = ((y[:, gaussian] - eta[:, gaussian]).T.dot(m[:, l]) - sum(v[:, l]) * a[l, gaussian]) / vgauss[gaussian]
            # accu_grad_a[l, :] = decay * accu_grad_a[l, :] + (1 - decay) * grad_a ** 2
            # neg_diag_Ha[family] = sum(lam * ((m[:, l, newaxis] + outer(v[:, l], a[l, family])) ** 2), axis=0) + \
            #               lam.T.dot(v[:, l])
            # neg_diag_Ha[gaussian] = sum(m[:, l] ** 2 + v[:, l]) / vgauss[gaussian]
            # delta_a = grad_a / (sqrt(eps + accu_grad_a[l, :]) + neg_diag_Ha)
            # a[l, :] += delta_a

        # estimate coefficients
        eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
        lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
        vhat = var(y - eta, axis=0, ddof=0)
        for n in range(N):
            if family[n] == 'poisson':
                va = v * a[:, n]  # (T, L)
                wv = diag(lam[:, n].dot(v))
                grad_a = m.T.dot(y[:, n]) - (m + va).T.dot(lam[:, n])

                # lb1 = elbo(y, h, family, chol, m, w, v, a2, b, vhat)
                # a2[:, n] += 1e-10 * grad_a
                # lb2 = elbo(y, h, family, chol, m, w, v, a2, b, vhat)
                # if lb2 < lb1:
                #     print('Inc a[{}] tiny grad = {}'.format(n, lb2 - lb1))

                neghess_a = (m + va).T.dot(lam[:, n, newaxis] * (m + va)) + wv
                delta_a = linalg.solve(neghess_a, grad_a, sym_pos=True)
                a[:, n] += delta_a

                # b Newton Raphson
                H = h[n, :]
                z = H.dot(b[:, n]) + (y[:, n] - lam[:, n]) / lam[:, n]
                b[:, n] = linalg.solve(H.T.dot(lam[:, n, newaxis] * H), H.T.dot(lam[:, n] * z), sym_pos=True)

                # grad_a = (y[:, n] - lam[:, n]).dot(m) - lam[:, n].dot(v * a[:, n])
                # for t in range(T):
                #     neghess_a += lam[t, n] * (outer(m[t, :] + v[t, :] * a[:, n], m[t, :] + v[t, :] * a[:, n]) +
                #                               diag(v[t, :]))
                # accu_grad_a[:, n] = accumulate(accu_grad_a[:, n], grad_a, decay)
                # delta_a = linalg.solve(neghess_a + diag(sqrt(eps + accu_grad_a[:, n])), grad_a, sym_pos=True)
                # a[:, n] += delta_a
            else:
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                a[:, n] = linalg.solve(m.T.dot(m) + diag(sum(v, axis=0)), m.T.dot(y[:, n] - h[n, :].dot(b[:, n])),
                                       sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, n] = linalg.solve(h[n, :].T.dot(h[n, :]), h[n, :].T.dot(y[:, n] - m.dot(a[:, n])), sym_pos=True)

        # update w
        for _ in range(3):
            eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
            lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
            vhat = var(y - eta, axis=0, ddof=0)
            for l in range(L):
                U = empty((T, N), dtype=float)
                U[:, poisson] = lam[:, poisson]
                U[:, gaussian] = 1 / vhat[gaussian]
                w[:, l] = U.dot(a[l, :] ** 2)
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

        good_m[:] = m
        good_a[:] = a
        good_b[:] = b

        i += 1

    stop = timeit.default_timer()

    lv = empty((L, T, cholrank), dtype=float)
    for l in range(L):
        G = chol[l, :]
        GTWG = G.T.dot(w[:, l].reshape((T, 1)) * G)

        A = eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))  # A should be pd but numerically not
        eigval, eigvec = linalg.eigh(A)
        eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
        lv[l, :] = G.dot(eigvec.dot(diag(sqrt(eigval))))

    return lbound[:i], m, lv, a, b, stop - start, converged
