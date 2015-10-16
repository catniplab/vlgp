import timeit
import numpy as np
from scipy import linalg
from util import selfh


def firingrate(h, m, v, a, b, lb=-30, ub=30):
    L, N = a.shape
    eta_x = m.dot(a) + 0.5 * v.dot(a ** 2)
    for n in range(N):
        eta_x[:, n] += h[:, n, :].dot(b[:, n])
    np.clip(eta_x, lb, ub, out=eta_x)
    return np.exp(eta_x)


def elbo(y, h, m, v, a, b, chol):
    """
    Evidence Lower Bound
    :param y: spike trains
    :param h: regressors
    :param m: posterior mean
    :param a: alpha
    :param b: beta
    :param chol: incomplete cholesky decomposition of prior covariance
    :param v: temporal slices of posterior variance
    :return: lower bound
    """
    N = y.shape[1]
    L, T, k = chol.shape
    eyek = np.identity(k)

    eta = m.dot(a)
    for n in range(N):
        eta[:, n] += h[:, n, :].dot(b[:, n])
    lam = firingrate(h, m, v, a, b)
    lb = np.sum(y * eta - lam)

    for l in range(L):
        G = chol[l, :]
        w = lam.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = G.T.dot(w * G)
        # mdivS = linalg.pinv2(G).dot(m[:, l])
        mdivS = linalg.lstsq(G, m[:, l])[0]
        trace = (T - np.trace(GTWG) + np.trace(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))
        lndet = np.linalg.slogdet(eyek - GTWG + GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]

        lb += -0.5 * np.inner(mdivS, mdivS) - 0.5 * trace + 0.5 * lndet
        # lb += 0.5 * lndet
    return lb


def train(y, p, chol, a0=None, b0=None, m0=None, niter=50, tol=1e-5, verbose=True):
    """
    :param y: (T, N), y trains
    :param p: order of regression
    :param prior_mean: (T, L), prior mean
    :param prior_var: (L,), prior variance
    :param prior_scale: (L,), prior inverse of squared lengthscale
    :param a0: (L, N), initial value of a
    :param b0: (N, intercept + p * N), initial value of b
    :param m0: (T, L), initial value of posterior mean
    :param fixalpha: bool, switch of not train a
    :param fixbeta: bool, switch of not train b
    :param fixpostmean: bool, switch of not train posterior mean
    :param anorm: norm constraint of a
    :param intercept: bool, include intercept term or not
    :param hyper: train hyperparameters or not
    :param control: control params
    :return:
        m: posterior mean
        post_cov: posterior covariance
        b: coefficient of h
        a: coefficient of latent
        a0: initial value of a
        b0: initial value of b
        lbound: array of lower bounds
        elapsed:
        converged:
    """

    #################################################
    def updatev():
        lam = firingrate(h, m, v, a, b)
        for l in range(L):
            G = chol[l, :]
            w = lam.dot(a[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)
    ###################################################

    # dimensions
    L, T, k = chol.shape
    N = y.shape[1]
    L = m0.shape[1]
    eyek = np.identity(k)

    # temporal slice of v
    v = np.ones((T, L)) * chol[:, 0, 0]

    # read-only variables, protection from unexpected assignment
    y.setflags(write=0)

    # construct makeregressor
    h = selfh(y, p)
    h.setflags(write=0)

    # initialize args
    # make a copy to avoid changing initial values
    if m0 is None:
        m = np.zeros((T, L), dtype=float)
    else:
        m = m0.copy()

    if a0 is None:
        a0 = np.random.randn(L, N)
    a = a0.copy()

    b = np.empty((1 + p, N), dtype=float)
    for n in range(N):
        b[:, n] = linalg.lstsq(h[:, n, :], y[:, n])[0]

    # valid values of parameters from previous iteration
    good_a = a.copy()
    good_b = b.copy()
    good_m = m.copy()

    # adagrad
    decay = 0.9
    eps = 1e-6
    accu_grad_a = np.zeros_like(a)

    # Optimization
    updatev()
    # initialize lower bound
    lbound = np.full(niter, np.finfo(float).min)
    lbound[0] = elbo(y, h, m, v, a, b, chol)
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(it))

        ###########
        # weights #
        ###########
        lam = firingrate(h, m, v, a, b)
        for n in range(N):
            grad_b = h[:, n, :].T.dot(y[:, n] - lam[:, n])
            neg_hess_b = h[:, n, :].T.dot((h[:, n, :].T * lam[:, n]).T)
            delta_b = linalg.lstsq(neg_hess_b, grad_b)[0]
            b[:, n] += delta_b
        updatev()

        #############
        # posterior #
        #############
        good_elbo = elbo(y, h, m, v, a, b, chol)
        lam = firingrate(h, m, v, a, b)
        for l in range(L):
            G = chol[l]
            w = lam.dot(a[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)

            u = (y - lam).dot(a[l, :])
            # grad_m = u - np.dot(linalg.pinv2(G).T, np.dot(linalg.pinv2(G), m[:, l]))
            # grad_m = u - linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]
            # accu_grad_m[:, l] = decay * accu_grad_m[:, l] + (1 - decay) * grad_m ** 2

            u2 = G.dot(G.T.dot(u)) - m[:, l]
            delta_m = u2 - G.dot((w * G).T.dot(u2)) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (w * G).T.dot(u2),
                                                                                  sym_pos=True)))
            m[:, l] = good_m[:, l] + delta_m
            m[:, l] -= np.mean(m[:, l])
            m[:, l] /= linalg.norm(m[:, l], ord=np.inf)
            lb = elbo(y, h, m, v, a, b, chol)
            if np.isfinite(lb) and lb > good_elbo:
                # Newton step
                good_elbo = lb
            else:
                # Gradient step
                # m[:, l] = good_m[:, l] + grad_m / np.sqrt(eps + accu_grad_m[:, l])
                # m[:, l] -= np.mean(m[:, l])
                # a[l, :] = good_a[l, :] * linalg.norm(m[:, l], ord=np.inf)
                # m[:, l] /= linalg.norm(m[:, l], ord=np.inf)
                m[:, l] = good_m[:, l]

            grad_a = (y - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * a[l, :]
            accu_grad_a[l, :] = decay * accu_grad_a[l, :] + (1 - decay) * grad_a ** 2
            # delta_a = np.sqrt(eps + accu_delta_a[l, :]) / np.sqrt(eps + accu_grad_a[l, :]) * grad_a
            # accu_delta_a[l, :] = decay * accu_delta_a[l, :] + (1 - decay) * delta_a ** 2
            # a[l, :] += 100 * delta_a
            neg_diag_Ha = np.sum(lam * ((m[:, l, np.newaxis] + np.outer(v[:, l], a[l, :])) ** 2), axis=0) + \
                          lam.T.dot(v[:, l])
            delta_a = grad_a / (np.sqrt(eps + accu_grad_a[l, :]) + neg_diag_Ha)
            a[l, :] += delta_a
        updatev()

        # store lower bound
        lbound[it] = elbo(y, h, m, v, a, b, chol)

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
                  'change in m = {:.10f}\n'.format(lbound[it], lbound[it] - lbound[it - 1],
                                                   timeit.default_timer() - iter_start,
                                                   change_a, change_b, change_m))

        if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]):
            converged = True

        # store current iteration
        good_a[:] = a
        good_b[:] = b
        good_m[:] = m

        it += 1

    stop = timeit.default_timer()
    return lbound[:it], m, a, b, stop - start, converged
