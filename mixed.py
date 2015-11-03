import timeit
import numpy as np
from scipy import linalg
from util import selfhistory
from numpy import identity, diag, eye, dot, einsum, inner, outer, trace, exp, log, sum, mean, var, min, max, abs, sqrt
from numpy import empty, empty_like, full, full_like, zeros, zeros_like, ones, ones_like, newaxis, tile
from numpy import inf, finfo, PINF, NINF

LB = -20
UB = 20


def llh(y, h, family, chol, m, v, a, b, vhat):
    L, T, k = chol.shape

    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))

    lpois = sum(y[:, family == 'poisson'] * eta[:, family == 'poisson'] - lam[:, family == 'poisson'])

    lgauss = - 0.5 * sum((y[:, family == 'gaussian'] - eta[:, family == 'gaussian']) ** 2 / vhat[family == 'gaussian'] +
                         v.dot(a[:, family == 'gaussian'] ** 2))

    lb = lpois + lgauss
    return lb


def elbo(y, h, family, chol, m, v, a, b, vhat):
    L, T, k = chol.shape
    N = y.shape[1]
    eyek = identity(k)

    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))

    lpois = sum(y[:, family == 'poisson'] * eta[:, family == 'poisson'] - lam[:, family == 'poisson'])

    lgauss = - 0.5 * sum(((y[:, family == 'gaussian'] - eta[:, family == 'gaussian']) ** 2 +
                         v.dot(a[:, family == 'gaussian'] ** 2)) / vhat[family == 'gaussian'])

    lb = lpois + lgauss

    for l in range(L):
        G = chol[l, :]
        U = empty((T, N), dtype=float)
        U[:, family == 'poisson'] = lam[:, family == 'poisson']
        U[:, family == 'gaussian'] = 1 / vhat[family == 'gaussian']
        w = U.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        A = GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))
        m_div_G = linalg.lstsq(G, m[:, l])[0]
        tr = T - trace(GTWG) + trace(A)
        lndet = np.linalg.slogdet(eyek - GTWG + A)[1]

        lb += -0.5 * inner(m_div_G, m_div_G) - 0.5 * tr + 0.5 * lndet

    return lb


def accumulate(accu, grad, decay):
    return decay * accu + (1 - decay) * grad ** 2


def residual(y, family, h, m, v, a, b, vhat):
    res = empty_like(y, dtype=float)
    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
    res[:, family == 'poisson'] = y[:, family == 'poisson'] - lam[:, family == 'poisson']
    res[:, family == 'gaussian'] = (y[:, family == 'gaussian'] - eta[:, family == 'gaussian']) / vhat[family == 'gaussian']
    return res


def train(y, family, p, chol, m0=None, a0=None, b0=None, niter=50, tol=1e-4, decay=0.9, eps=1e-6, verbose=True):
    L, T, cholrank = chol.shape
    N = y.shape[1]
    eyek = identity(cholrank)

    y0 = zeros(N, dtype=float)
    y0[family == 'gaussian'] = mean(y[:, family == 'gaussian'], axis=0)
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
    b2 = b.copy()
    m2 = m.copy()

    good_m = m.copy()
    good_a = a.copy()
    good_b = b.copy()

    v = zeros_like(m, dtype=float)

    vhat = var(y - einsum('ijk, ki->ji', h, b) + m.dot(a), axis=0, ddof=0)
    lbound = full(niter, fill_value=finfo(float).min, dtype=float)
    lbound[0] = elbo(y, h, family, chol, m, v, a, b, vhat)

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
        eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
        lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
        vhat = var(y - eta, axis=0, ddof=0)

        m2[:] = good_m

        for l in range(L):
            # m
            G = chol[l]
            grad_l = (y[:, family == 'poisson'] - lam[:, family == 'poisson']).dot(a[l, family == 'poisson']) + \
                     ((y[:, family == 'gaussian'] - eta[:, family == 'gaussian']) / vhat[family == 'gaussian']).dot(a[l, family == 'gaussian'])
            grad_m = grad_l - linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]
            # accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m, decay)
            accu_grad_m[:, l] = accumulate(accu_grad_m[:, l], grad_m / linalg.norm(grad_m, ord=inf), decay)

            U = empty((T, N), dtype=float)
            U[:, family == 'poisson'] = lam[:, family == 'poisson']
            U[:, family == 'gaussian'] = 1 / vhat[family == 'gaussian']
            # w = (U.dot(a[l, :] ** 2)).reshape((T, 1))  # adjusted by adagrad
            w = (U.dot(a[l, :] ** 2) + sqrt(eps + accu_grad_m[:, l])).reshape((T, 1))  # adjusted by adagrad
            GTWG = G.T.dot(w * G)

            R = residual(y, family, h, m, v, a, b, vhat)

            u = G.dot(G.T.dot(R.dot(a[l, :]))) - m[:, l]
            delta_m = u - G.dot((w * G).T.dot(u)) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (w * G).T.dot(u),
                                                                                  sym_pos=True)))
            # print('||grad m[{}]|| = {}'.format(l, linalg.norm(grad_m)))
            # print('||delta m[{}]|| = {}'.format(l, linalg.norm(delta_m)))

            lb1 = elbo(y, h, family, chol, good_m, v, a, b, vhat)
            m2[:, l] = good_m[:, l] + 1e-10 * grad_m / linalg.norm(grad_m, ord=inf)
            lb2 = elbo(y, h, family, chol, m2, v, a, b, vhat)

            # print(linalg.norm(m2))
            # print(linalg.norm(good_m))

            # lh1 = llh(y, h, family, chol, good_m, v, a, b, vhat)
            # m2[:, l] = good_m[:, l] + 0 * grad_l / linalg.norm(grad_l, ord=inf)
            # lh2 = llh(y, h, family, chol, m2, v, a, b, vhat)

            print('Inc[{}] tiny grad = {}'.format(l, lb2 - lb1))
            # print('Inc[{}] lh {}'.format(l, lh2 - lh1))

            m[:, l] = good_m[:, l] + delta_m

            # lb2 = elbo(y, h, family, chol, m, v, a, b, vhat)
            # print('Inc[{}] before constraint = {}'.format(l, lb2 - lbound[i - 1]))

            m[:, l] -= mean(m[:, l])
            scale = linalg.norm(m[:, l], ord=inf)
            # a[l, :] *= scale
            m[:, l] /= scale

            # grad_a = empty(N, dtype=float)
            # neg_diag_Ha = empty_like(grad_a)
            # grad_a[family] = (y[:, family] - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * a[l, family]
            # grad_a[family == 'gaussian'] = ((y[:, family == 'gaussian'] - eta[:, family == 'gaussian']).T.dot(m[:, l]) - sum(v[:, l]) * a[l, family == 'gaussian']) / vgauss[family == 'gaussian']
            # accu_grad_a[l, :] = decay * accu_grad_a[l, :] + (1 - decay) * grad_a ** 2
            # neg_diag_Ha[family] = sum(lam * ((m[:, l, newaxis] + outer(v[:, l], a[l, family])) ** 2), axis=0) + \
            #               lam.T.dot(v[:, l])
            # neg_diag_Ha[family == 'gaussian'] = sum(m[:, l] ** 2 + v[:, l]) / vgauss[family == 'gaussian']
            # delta_a = grad_a / (sqrt(eps + accu_grad_a[l, :]) + neg_diag_Ha)
            # a[l, :] += delta_a

        # estimate coefficients
        eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
        lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))
        vhat = var(y - eta, axis=0, ddof=0)
        # v
        for l in range(L):
            G = chol[l, :]
            U = empty((T, N), dtype=float)
            U[:, family == 'poisson'] = lam[:, family == 'poisson']
            U[:, family == 'gaussian'] = 1 / vhat[family == 'gaussian']
            w = U.dot(a[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            v[:, l] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)

        for n in range(N):
            # neghess_b = zeros((1 + p, 1 + p), dtype=float)
            if family[n] == 'poisson':
                # grad_b = (y[:, n] - lam[:, n]).dot(h[n, :])
                # for t in range(T):
                #     neghess_b += lam[t, n] * outer(h[n, t, :], h[n, t, :])
                # accu_grad_b[:, n] = accumulate(accu_grad_b[:, n], grad_b, decay)
                # delta_b = linalg.solve(neghess_b + diag(sqrt(eps + accu_grad_b[:, n])), grad_b, sym_pos=True)
                # b[:, n] += delta_b
                H = h[n, :]
                z = H.dot(b[:, n]) + (y[:, n] - lam[:, n]) / lam[:, n]
                b[:, n] = linalg.solve(H.T.dot(lam[:, n, newaxis] * H), H.T.dot(lam[:, n] * z), sym_pos=True)

                grad_a = m.T.dot(y[:, n] - lam[:, n]) - diag(lam[:, n].dot(v)).dot(a[:, n])
                # a2[:, n] = good_a[:, n] + 1e-10 * grad_a
                # lb2 = elbo(y, h, family, chol, m, v, a2, b, vhat)
                # print('Inc[{}] tiny grad a = {}'.format(n, lb2 - lbound[i - 1]))

                neghess_a = m.T.dot(lam[:, n, newaxis] * m) + diag(lam[:, n].dot(v))
                delta_a = linalg.solve(neghess_a, grad_a, sym_pos=True)
                a[:, n] += delta_a

                # grad_a = (y[:, n] - lam[:, n]).dot(m) - lam[:, n].dot(v * a[:, n])
                # for t in range(T):
                #     neghess_a += lam[t, n] * (outer(m[t, :] + v[t, :] * a[:, n], m[t, :] + v[t, :] * a[:, n]) +
                #                               diag(v[t, :]))
                # accu_grad_a[:, n] = accumulate(accu_grad_a[:, n], grad_a, decay)
                # delta_a = linalg.solve(neghess_a + diag(sqrt(eps + accu_grad_a[:, n])), grad_a, sym_pos=True)
                # a[:, n] += delta_a
            else:
                # least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, n] = linalg.solve(h[n, :].T.dot(h[n, :]), h[n, :].T.dot(y[:, n] - m.dot(a[:, n])), sym_pos=True)

                # grad_b = ((y[:, n] - eta[:, n]) / vhat[n]).dot(h[n, :])
                # for t in range(T):
                #     neghess_b += outer(h[n, t, :], h[n, t, :]) / vhat[n]

                # least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                a[:, n] = linalg.solve(m.T.dot(m) + diag(sum(v, axis=0)), m.T.dot(y[:, n] - h[n, :].dot(b[:, n])),
                                       sym_pos=True)

                # grad_a = ((y[:, n] - eta[:, n]).dot(m) - sum(v * a[:, n], axis=0)) / vhat[n]
                # for t in range(T):
                #     neghess_a += (outer(m[t, :], m[t, :]) + diag(v[t, :])) / vhat[n]

        lbound[i] = elbo(y, h, family, chol, m, v, a, b, vhat)

        # check convergence
        change_a = max(abs(good_a - a))
        change_b = max(abs(good_b - b))
        change_m = max(abs(good_m - m))

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'
                  'change in a = {:.10f}\n'
                  'change in b = {:.10f}\n'
                  'change in m = {:.10f}\n'.format(lbound[i], lbound[i] - lbound[i - 1],
                                                   timeit.default_timer() - iter_start,
                                                   change_a, change_b, change_m))

        if abs(lbound[i] - lbound[i - 1]) < tol * abs(lbound[i - 1]):
            converged = True

        good_m[:] = m
        good_a[:] = a
        good_b[:] = b

        i += 1

    stop = timeit.default_timer()

    lv = empty((L, T, cholrank), dtype=float)
    eta = einsum('ijk, ki->ji', h, b) + m.dot(a)
    lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(LB, UB))

    for l in range(L):
        G = chol[l, :]
        u = empty((T, N), dtype=float)
        u[:, family == 'poisson'] = lam[:, family == 'poisson']
        u[:, family == 'gaussian'] = 1 / vhat[family == 'gaussian']
        w = u.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)

        A = eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))  # A should be pd but numerically not
        eigval, eigvec = linalg.eigh(A)
        eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
        lv[l, :] = G.dot(eigvec.dot(diag(sqrt(eigval))))

    return lbound[:i], m, lv, a, b, stop - start, converged
