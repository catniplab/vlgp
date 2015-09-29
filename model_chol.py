__author__ = 'yuan'
import timeit

import numpy as np
from scipy import linalg

from util import makeregressor


def incchol(n, omega, k, tol=1e-10):
    """
    Incomplete Cholesky decomposition for squared exponential covariance
    :param n: size of covariance matrix (n, n)
    :param omega: inverse of 2 * squared lengthscale
    :param k: number of columns of decomposition
    :return: (n, m) matrix
    """
    # x = np.arange(n)
    x = np.linspace(0, 1, n)
    diagG = np.ones(n, dtype=float)
    pvec = np.arange(n, dtype=int)
    i = 0
    g = np.zeros((n, k), dtype=float)
    while i < k and np.sum(diagG[i:]) > tol:
        if i > 0:
            jast = np.argmax(diagG[i:])
            jast += i
            pvec[i], pvec[jast] = pvec[jast].copy(), pvec[i].copy()
            g[jast, :i + 1], g[i, :i + 1] = g[i, :i + 1].copy(), g[jast, :i + 1].copy()
        else:
            jast = 0

        g[i, i] = np.sqrt(diagG[jast])
        newAcol = np.exp(- omega * (x[pvec[i + 1:]] - x[pvec[i]]) ** 2)
        g[i + 1:, i] = (newAcol - np.dot(g[i + 1:, :i], g[i, :i].T)) / g[i, i]
        diagG[i + 1:] = 1 - np.sum((g[i + 1:, :i + 1]) ** 2, axis=1)

        i += 1
    return g[np.argsort(pvec), :]


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
    eta_x = eta + 0.5 * v.dot(a ** 2)
    np.clip(eta_x, 10 * np.finfo(np.float).eps, np.log(np.finfo(np.float).max / 10), out=eta_x)
    lam = np.exp(eta_x)
    lb = np.sum(y * eta - lam)

    for l in range(L):
        G = chol[l, :]
        w = lam.dot(a[l, :] ** 2).reshape((T, 1))
        GTWG = np.dot(G.T, w * G)
        mdivS = linalg.pinv2(G).dot(m[:, l])
        trace = (T - np.trace(GTWG) + np.trace(np.dot(GTWG, linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))
        lndet = np.linalg.slogdet(eyek - GTWG + np.dot(GTWG, linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))[1]

        lb += -0.5 * np.inner(mdivS, mdivS) - 0.5 * trace + 0.5 * lndet
        # lb += 0.5 * lndet
    return lb


def train(spike, p, prior_var, prior_scale, a0=None, b0=None, m0=None, normofalpha=1.0,
          hyper=False, fixalpha=False, fixbeta=False, fixpostmean=False, kchol=9, control=None):
    """
    :param spike: (T, N), spike trains
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
    :param normofalpha: norm constraint of alpha
    :param intercept: bool, include intercept term or not
    :param hyper: train hyperparameters or not
    :param control: control params
    :return:
        post_mean: posterior mean
        post_cov: posterior covariance
        beta: coefficient of regressor
        alpha: coefficient of latent
        a0: initial value of alpha
        b0: initial value of beta
        lbound: array of lower bounds
        elapsed:
        converged:
    """

    def updatev(r=2):
        for _ in range(r):
            lam = firingrate()
            for l in range(L):
                G = prior_chol[l, :]
                w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
                GTWG = np.dot(G.T, w * G)
                v[:, l] = np.sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyek + GTWG, GTWG, sym_pos=True)))),
                                 axis=1)

    def makechol():
        for l in range(L):
            prior_chol[l, :] = incchol(T, prior_scale[l], kchol) * np.sqrt(prior_var[l])

    def firingrate():
        eta_x = regressor.dot(beta) + post_mean.dot(alpha) + 0.5 * v.dot(alpha ** 2)
        np.clip(eta_x, eps, 30, out=eta_x)
        return np.exp(eta_x)
    ###################################################

    # epsilon
    eps = 2 * np.finfo(np.float).eps

    # dimensions
    T, N = spike.shape
    L = len(prior_var)

    eyek = np.identity(kchol)

    # control
    maxiter = control['max iteration']
    fpinter = control['fixed-point iteration']
    tol = control['tol']
    verbose = control['verbose']

    # hyperparameters
    prior_var = prior_var.copy()
    prior_scale = prior_scale.copy()

    # incomplete cholesky decomposition of prior covariance matrix
    prior_chol = np.empty((L, T, kchol), dtype=float)
    makechol()

    # temporal slice of v
    v = np.ones((T, L)) * prior_var

    # read-only variables, protection from unexpected assignment
    spike.setflags(write=0)

    # construct makeregressor
    regressor = makeregressor(spike, p, intercept=True)
    regressor.setflags(write=0)

    # initialize args
    # make alpha copy to avoid changing initial values
    if m0 is None:
        post_mean = np.zeros((T, L), dtype=float)
    else:
        post_mean = m0.copy()

    if a0 is None:
        a0 = np.random.randn(L, N)
        a0 /= linalg.norm(a0) / normofalpha
    alpha = a0.copy()

    if b0 is None:
        b0 = linalg.lstsq(regressor, spike)[0]
    beta = b0.copy()

    # valid values of parameters from previous iteration
    good_alpha = alpha.copy()
    good_beta = beta.copy()
    good_post_mean = post_mean.copy()
    good_var = prior_var.copy()
    good_scale = prior_scale.copy()

    # temporary storage for recovery
    new_beta = beta.copy()
    new_alpha = alpha.copy()
    new_m = post_mean.copy()
    last_var = prior_var.copy()
    new_scale = prior_scale.copy()
    new_chol = prior_chol.copy()

    stepsize_alpha = np.ones(N)
    stepsize_beta = np.ones(N)
    stepsize_post_mean = np.ones(L)

    direction_w = np.ones(L) * 1.1

    deflation = 0.5
    inflation = 1.5
    thld = 0.5

    # Optimization
    updatev()
    # initialize lower bound
    lbound = np.full(maxiter, np.finfo(float).min)
    lbound[0] = elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it < maxiter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(it))

        if not fixbeta:
            new_beta[:] = beta
            good_elbo = elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)
            for n in range(N):
                lam = firingrate()
                grad_b = np.dot(regressor.T, spike[:, n] - lam[:, n])
                neg_hess_b = np.dot(regressor.T, (regressor.T * lam[:, n]).T)
                if linalg.norm(grad_b, ord=np.inf) < eps:
                    break
                try:
                    delta_b = stepsize_beta[n] * linalg.solve(neg_hess_b, grad_b, sym_pos=True)
                except linalg.LinAlgError as e:
                    print('beta', e)
                    continue
                elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)
                new_beta[:, n] = beta[:, n] + delta_b
                lb = elbo(spike, regressor, post_mean, alpha, new_beta, prior_chol, v)
                predicted = thld * np.inner(grad_b, delta_b)
                if np.isnan(lb) or lb < good_elbo:
                    # Decrease the stepsize if the lower bound decreases.
                    # Add a small positive number to prevent becoming 0.
                    stepsize_beta[n] *= deflation
                    stepsize_beta[n] += eps
                else:
                    if lb - good_elbo > predicted:
                        # Increase the stepsize if the real increment is more than expected.
                        stepsize_beta[n] *= inflation
                    good_elbo = lb
                    beta[:, n] = new_beta[:, n]
            updatev()

        if not fixalpha:
            new_alpha[:] = alpha
            good_elbo = elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)
            for l in range(L):
                lam = firingrate()
                grad_a = np.dot((spike - lam).T, post_mean[:, l]) \
                         - np.dot(lam.T, v[:, l]) * alpha[l, :]
                neg_hess_a = np.diag(np.dot(lam.T, post_mean[:, l] ** 2) +
                                     2 * np.dot(lam.T, post_mean[:, l] * v[:, l]) * alpha[l, :] +
                                     np.dot(lam.T, v[:, l] ** 2) * alpha[l, :] ** 2 +
                                     np.dot(lam.T, v[:, l]))
                if linalg.norm(grad_a, ord=np.inf) < eps:
                    break
                try:
                    delta_a = stepsize_alpha[l] * linalg.solve(neg_hess_a, grad_a, sym_pos=True)
                except linalg.LinAlgError as e:
                    print('alpha', e)
                    continue
                new_alpha[l, :] = alpha[l, :] + delta_a
                new_alpha[l, :] /= linalg.norm(new_alpha[l, :]) / normofalpha
                lb = elbo(spike, regressor, post_mean, new_alpha, beta, prior_chol, v)
                predicted = thld * np.inner(grad_a, delta_a)
                if np.isnan(lb) or lb < good_elbo:
                    stepsize_alpha[l] *= deflation
                    stepsize_alpha[l] += eps
                    # alpha[l, :] = new_alpha[l, :]
                    # rate[:] = last_rate[:]
                else:
                    if lb - good_elbo > predicted:
                        stepsize_alpha[l] *= inflation
                    good_elbo = lb
                    alpha[l, :] = new_alpha[l, :]
            updatev()

        # posterior mean
        if not fixpostmean:
            new_m[:] = post_mean
            good_elbo = elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)
            for l in range(L):
                lam = firingrate()
                G = prior_chol[l]
                w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
                GTWG = np.dot(G.T, w * G)

                u = np.dot(spike - lam, alpha[l, :])
                grad_m = u - np.dot(linalg.pinv2(G).T, np.dot(linalg.pinv2(G), post_mean[:, l]))

                u2 = G.dot(np.dot(G.T, u)) - post_mean[:, l]
                delta_m = u2 - G.dot(np.dot((w * G).T, u2)) + \
                          G.dot(GTWG.dot(linalg.solve(eyek + GTWG, np.dot((w * G).T, u2), sym_pos=True)))
                delta_m *= stepsize_post_mean[l]
                new_m[:, l] = post_mean[:, l]
                new_m[:, l] += delta_m
                new_m[:, l] -= np.mean(new_m[:, l])
                lb = elbo(spike, regressor, new_m, alpha, beta, prior_chol, v)
                predicted = thld * np.inner(grad_m, np.squeeze(delta_m))
                if np.isnan(lb) or lb < good_elbo:
                    stepsize_post_mean[l] *= deflation
                    stepsize_post_mean[l] += eps
                else:
                    if lb - good_elbo > predicted:
                        stepsize_post_mean[l] *= inflation
                    good_elbo = lb
                    post_mean[:, l] = new_m[:, l]
            updatev()

        if hyper and it % 5 == 0:
            for l in range(L):
                lam = firingrate()
                last_var[l] = prior_var[l]
                for _ in range(fpinter):
                    G = prior_chol[l]
                    w = lam.dot(alpha[l, :] ** 2).reshape((T, 1))
                    GTWG = np.dot(G.T, w * G)
                    mdivS = linalg.pinv2(G).dot(post_mean[:, l])
                    new_var = prior_var[l] * (np.inner(mdivS, mdivS) + \
                                              (T - np.trace(GTWG) + np.trace(
                                                  np.dot(GTWG, linalg.solve(eyek + GTWG, GTWG, sym_pos=True))))) / T
                    prior_var[l] = new_var if new_var > 0 else prior_var[l] / 2
                if verbose:
                    print('prior variance[{:d}]: {:.5f} -> {:.5f}'.format(l, last_var[l], prior_var[l]))
            makechol()
            # updatev()

            # grid_w = np.array(list(itertools.combinations_with_replacement(np.logspace(-1, 3, 5), 2)))

            # lb = np.empty(grid_w.shape[0], dtype=float)
            # for i, row in enumerate(grid_w):
            #     for l in range(L):
            #         new_chol[l, :] = incchol(T, row[l], kchol) * np.sqrt(prior_var[l])
            #     lb[i] = elbo(spike, regressor, post_mean, alpha, beta, new_chol, v)
            #     # print(row, lb[i])
            #
            # prior_scale[:] = grid_w[lb.argmax(), :]
            # low = -2
            # high = 6
            # new_chol[:] = prior_chol
            # for l in range(L):
            #     # grid = cartesian([omega] * L)
            #     omega = np.append(np.logspace(low, high, num=high - low + 1), prior_scale[l])
            #     lb = np.full(omega.shape[0], fill_value=np.NINF)
            #     for i, row in enumerate(omega):
            #         new_chol[l, :] = incchol(T, row, kchol) * np.sqrt(prior_var[l])
            #         lb[i] = elbo(spike, regressor, post_mean, alpha, beta, new_chol, v)
            #     if verbose:
            #         print(np.column_stack((omega, lb)))
            #     best = omega[lb.argmax()]
            #     if verbose:
            #         print('omega[{}]: {} -> {}'.format(l, prior_scale[l], best))
            #     prior_scale[l] = best
            # makechol()

        # store lower bound
        lbound[it] = elbo(spike, regressor, post_mean, alpha, beta, prior_chol, v)

        # check convergence
        chg_alpha = 0.0 if fixalpha else np.max(np.abs(good_alpha - alpha))
        chg_beta = 0.0 if fixbeta else np.max(np.abs(good_beta - beta))
        chg_post_mean = 0.0 if fixpostmean else np.max(np.abs(good_post_mean - post_mean))
        chg_variance = np.max(np.abs(good_var - prior_var)) if hyper else 0.0
        chg_scale = np.max(np.abs(good_scale - prior_scale)) if hyper else 0.0

        # converged if the change in ELBO is relatively smaller than a tolerance
        if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]):
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
        good_post_mean[:] = post_mean
        good_var[:] = prior_var
        good_scale[:] = prior_scale

        it += 1

    stop = timeit.default_timer()

    return lbound[:it], post_mean, alpha, beta, prior_var, prior_scale, a0, b0, stop - start, converged
