import timeit
import numpy as np
from scipy import linalg
from util import makeregressor
from la import ichol_gauss
from pylab import plot
__author__ = 'yuan'


def firingrate(h, m, v, a, b, lb=-30, ub=30):
    eta_x = h.dot(b) + m.dot(a) + 0.5 * v.dot(a ** 2)
    np.clip(eta_x, lb, ub, out=eta_x)
    return np.exp(eta_x)


def elbo(y, h, m, a, b, chol, v):
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
    # T, N = y.shape
    L, T, k = chol.shape
    eyek = np.identity(k)

    eta = h.dot(b) + m.dot(a)
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


def train(y, p, prior_var, prior_scale, a0=None, b0=None, m0=None, anorm=1.0,
          hyper=False, kchol=10, niter=50, tol=1e-5, verbose=True):
    """
    :param y: (T, N), y trains
    :param p: order of regression
    :param prior_mean: (T, L), prior mean
    :param prior_var: (L,), prior variance
    :param prior_scale: (L,), prior inverse of squared lengthscale
    :param a0: (L, N), initial value of alpha
    :param b0: (N, intercept + p * N), initial value of beta
    :param m0: (T, L), initial value of posterior mean
    :param fixalpha: bool, switch of not train alpha
    :param fixbeta: bool, switch of not train beta
    :param fixpostmean: bool, switch of not train posterior mean
    :param anorm: norm constraint of alpha
    :param intercept: bool, include intercept term or not
    :param hyper: train hyperparameters or not
    :param control: control params
    :return:
        m: posterior mean
        post_cov: posterior covariance
        beta: coefficient of h
        alpha: coefficient of latent
        a0: initial value of alpha
        b0: initial value of beta
        lbound: array of lower bounds
        elapsed:
        converged:
    """

    #################################################
    def updatev():
        lam = firingrate(h, m, v, alpha, beta)
        for l in range(L):
            G = prior_chol[l, :]
            w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)
            v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                             axis=1)

    # def tryv(m, a, b):
    #     out = np.empty_like(v)
    #     lam = firingrate(h, m, v, a, b)
    #     for l in range(L):
    #         G = prior_chol[l, :]
    #         w = lam.dot(a[l, :] ** 2).reshape((T, 1))
    #         GTWG = G.T.dot(w * G)
    #         out[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
    #                          axis=1)
    #     return out

    def makechol():
        for l in range(L):
            prior_chol[l, :] = ichol_gauss(T, prior_scale[l], kchol) * np.sqrt(prior_var[l])

    ###################################################

    # epsilon
    eps = 2 * np.finfo(np.float).eps

    # dimensions
    T, N = y.shape
    L = len(prior_var)

    eyek = np.identity(kchol)

    # hyperparameters
    prior_var = prior_var.copy()
    prior_scale = prior_scale.copy()

    # incomplete cholesky decomposition of prior covariance matrix
    prior_chol = np.empty((L, T, kchol), dtype=float)
    makechol()

    # temporal slice of v
    v = np.ones((T, L)) * prior_var

    # read-only variables, protection from unexpected assignment
    y.setflags(write=0)

    # construct makeregressor
    h = makeregressor(y, p, intercept=True)
    h.setflags(write=0)

    # initialize args
    # make alpha copy to avoid changing initial values
    if m0 is None:
        m = np.zeros((T, L), dtype=float)
    else:
        m = m0.copy()

    if a0 is None:
        a0 = np.random.randn(L, N)
        a0 /= linalg.norm(a0) / anorm
    alpha = a0.copy()

    if b0 is None:
        b0 = linalg.lstsq(h, y)[0]
    beta = b0.copy()

    # valid values of parameters from previous iteration
    good_alpha = alpha.copy()
    good_beta = beta.copy()
    good_m = m.copy()
    good_var = prior_var.copy()
    good_scale = prior_scale.copy()

    step_alpha = 1e-2
    step_beta = 1e-1
    step_m = 1e-2

    accu_grad_m = np.zeros_like(m)
    accu_grad_a = np.zeros_like(alpha)
    accu_grad_b = np.zeros_like(beta)

    # Optimization
    updatev()
    # initialize lower bound
    lbound = np.full(niter, np.finfo(float).min)
    lbound[0] = elbo(y, h, m, alpha, beta, prior_chol, v)
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(it))

        good_elbo = lbound[it - 1]
        #############
        # posterior #
        #############
        lam = firingrate(h, m, v, alpha, beta)
        for l in range(L):
            G = prior_chol[l]
            w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
            GTWG = G.T.dot(w * G)

            u = (y - lam).dot(alpha[l, :])
            # grad_m = u - np.dot(linalg.pinv2(G).T, np.dot(linalg.pinv2(G), m[:, l]))
            grad_m = u - linalg.lstsq(G.T, linalg.lstsq(G, m[:, l])[0])[0]

            u2 = G.dot(G.T.dot(u)) - m[:, l]
            delta_m = u2 - G.dot((w * G).T.dot(u2)) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, (w * G).T.dot(u2),
                                                                                  sym_pos=True)))

            accu_grad_m[:, l] = 0.95 * accu_grad_m[:, l] + 0.05 * grad_m ** 2
            delta_m /= (1e-6 + np.sqrt(accu_grad_m[:, l]))
            m[:, l] += step_m * delta_m
            # plot(m[:, l])
            lb = elbo(y, h, m, alpha, beta, prior_chol, v)
            m[:, l] -= np.mean(m[:, l])
            # predicted = thld * np.inner(grad_m, delta_m)
            if np.isfinite(lb) and lb > good_elbo:
                # if lb - good_elbo > predicted:
            #         step_m[l] *= up
                good_elbo = lb
            #     m[:, l] -= np.mean(m[:, l])
            else:
                m[:, l] = good_m[:, l]
            #     step_m[l] *= down
        # step_m.clip(10 * np.finfo(float).eps, np.finfo(float).max, out=step_m)
        updatev()

        ###########
        # weights #
        ###########
        lam = firingrate(h, m, v, alpha, beta)
        for l in range(L):
            grad_a = (y - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * alpha[l, :]
            # neg_hess_a = np.diag(lam.T.dot(m[:, l] ** 2) +
            #                      2 * lam.T.dot(m[:, l] * v[:, l]) * alpha[l, :] +
            #                      lam.T.dot(v[:, l] ** 2) * alpha[l, :] ** 2 +
            #                      lam.T.dot(v[:, l]))
            # neg_diag_Ha = lam.T.dot(m[:, l] ** 2) + 2 * lam.T.dot(m[:, l] * v[:, l]) * alpha[l, :] + lam.T.dot(
            #     v[:, l] ** 2) * alpha[l, :] ** 2 + lam.T.dot(v[:, l])
            # neg_diag_Ha = np.sum(lam * ((m[:, l, np.newaxis] + np.outer(v[:, l], alpha[l, :])) ** 2), axis=0) + \
            #               lam.T.dot(v[:, l])
            # delta_a = grad_a / (neg_diag_Ha.clip(0.5, np.PINF))
            accu_grad_a[l, :] = 0.95 * accu_grad_a[l, :] + 0.05 * grad_a ** 2
            adj_grad_a = grad_a / (1e-6 + np.sqrt(accu_grad_a[l, :]))
            alpha[l, :] += step_alpha * adj_grad_a
            alpha[l, :] /= linalg.norm(alpha[l, :]) / anorm
        updatev()

        lam = firingrate(h, m, v, alpha, beta)
        for n in range(N):
            grad_b = h.T.dot(y[:, n] - lam[:, n])
            # neg_hess_b = h.T.dot((h.T * lam[:, n]).T)
            # if linalg.norm(grad_b, ord=np.inf) < eps:
            #     break
            # try:
            #     delta_b = linalg.solve(neg_hess_b, grad_b, sym_pos=True)
            # except linalg.LinAlgError as e:
            #     print('beta', e)
            #     continue
            accu_grad_b[:, n] = 0.95 * accu_grad_b[:, n] + 0.05 * grad_b ** 2
            delta_b = grad_b / (1e-6 + np.sqrt(accu_grad_b[:, n]))
            # accu_delta_b[:, n] = 0.95 * accu_delta_b[:, n] + 0.05 * delta_b ** 2
            beta[:, n] += step_beta * delta_b
        updatev()

        # store lower bound
        lbound[it] = elbo(y, h, m, alpha, beta, prior_chol, v)

        # check convergence
        chg_alpha = np.max(np.abs(good_alpha - alpha))
        chg_beta = np.max(np.abs(good_beta - beta))
        chg_post_mean = np.max(np.abs(good_m - m))
        chg_variance = np.max(np.abs(good_var - prior_var)) if hyper else 0.0
        chg_scale = np.max(np.abs(good_scale - prior_scale)) if hyper else 0.0

        # converged if the change in ELBO is relatively smaller than a tolerance
        # if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]):
        #     converged = True

        if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]) and it % 5 == 0:
            converged = True

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'
                  'change in alpha = {:.10f}\n'
                  'change in beta = {:.10f}\n'
                  'change in posterior mean = {:.10f}\n'
                  'change in prior variance = {:.10f}\n'
                  'change in prior scale = {:.10f}'.format(lbound[it], lbound[it] - lbound[it - 1],
                                                           timeit.default_timer() - iter_start,
                                                           chg_alpha, chg_beta, chg_post_mean,
                                                           chg_variance, chg_scale))

        # store current iteration
        good_alpha[:] = alpha
        good_beta[:] = beta
        good_m[:] = m
        good_var[:] = prior_var
        good_scale[:] = prior_scale

        it += 1

    stop = timeit.default_timer()
    return lbound[:it], m, alpha, beta, prior_var, prior_scale, a0, b0, stop - start, converged
