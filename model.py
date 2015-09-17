import itertools
import timeit

import numpy as np
from scipy import linalg

from util import makeregressor, sqexpcov


def train(spike, p, prior_mean, prior_var, prior_scale,
                a0=None, b0=None, m0=None, V0=None,
                guardV=True, guardSigma=True,
                fixalpha=False, fixbeta=False, fixpostmean=False, fixpostcov=False, normofalpha=1.0, intercept=True,
                hyper=False,
                control=None):
    """
    :param spike: (T, N), spike trains
    :param p: order of regression
    :param prior_mean: (T, L), prior mean
    :param prior_var: (L,), prior variance
    :param prior_scale: (L,), prior inverse of squared lengthscale
    :param a0: (L, N), initial value of alpha
    :param b0: (N, intercept + p * N), initial value of beta
    :param m0: (T, L), initial value of posterior mean
    :param V0: (L, T, T), initial value of posterior covariance
    :param fixalpha: bool, switch of not train alpha
    :param fixbeta: bool, switch of not train beta
    :param fixpostmean: bool, switch of not train posterior mean
    :param fixpostcov: bool, switch of not train posterior covariance
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

    ##########################################################################
    # frequently used matrices
    def prior_cov(l):
        return prior_cor[l, :, :] * prior_var[l]

    def wroot(l):
        return np.sqrt(np.diag(np.dot(rate, alpha[l, :] ** 2)))

    def bmatrix(l):
        wsqrt = wroot(l)
        return eyeT + np.dot(wsqrt, np.dot(prior_cov(l), wsqrt))

    def invsigma_dot_v(l):
        bmat = bmatrix(l)
        wsqrt = wroot(l)
        return eyeT - np.dot(wsqrt, np.dot(linalg.solve(bmat, wsqrt), prior_cov(l)))

    # expected firing rate
    # fix insane value
    def saferate(t, n):
        eta_regress = np.dot(regressor[t, :], beta[:, n])
        eta_latent = np.dot(post_mean[t, :], alpha[:, n])
        eta_cov = 0.5 * np.sum(alpha[:, n] ** 2 * post_cov[:, t, t])
        eta = eta_regress + eta_latent + eta_cov
        srate = np.nan_to_num(np.exp(eta))
        return srate if srate > 0 else np.finfo(np.float).eps

    # ELBO
    def elbo():
        lb = np.sum(spike * (np.dot(regressor, beta) + np.dot(post_mean, alpha)) - rate)

        for l in range(L):
            d = post_mean[:, l] - prior_mean[:, l]
            w = np.diag(np.dot(rate, alpha[l, :] ** 2))
            # wsqrt = wroot(l)
            # bmat = bmatrix(l)
            isv = invsigma_dot_v(l)

            lb += -0.5 * np.dot(d, np.dot(linalg.pinvh(prior_cov(l)), d)) + \
                  -0.5 * np.trace(isv) + \
                  -0.5 * np.linalg.slogdet(eyeT + np.dot(prior_cov(l), w))[1]

        return lb

    def updaterate(rows, cols):
        # rate = E(E(spike|x))
        for row, col in itertools.product(rows, cols):
            rate[row, col] = saferate(row, col)

    def updatepostcov():
        for l in range(L):
            wsqrt = wroot(l)
            bmat = bmatrix(l)
            post_cov[l, :, :] = prior_cov(l) - np.dot(prior_cov(l), np.dot(wsqrt, np.dot(linalg.solve(bmat, wsqrt),
                                                                                         prior_cov(l))))
        updaterate(range(T), range(N))

    ###################################################

    # epsilon
    eps = 2 * np.finfo(np.float).eps

    # dimensions
    T, N = spike.shape
    _, L = prior_mean.shape

    eyeT = np.identity(T)

    # control
    maxiter = control['max iteration']
    fpinter = control['fixed-point iteration']
    tol = control['tol']
    verbose = control['verbose']

    # hyperparameters
    prior_var = prior_var.copy()
    prior_scale = prior_scale.copy()

    i, j = np.meshgrid(np.arange(T), np.arange(T))
    logcor = -(i - j) ** 2

    # prior_chol = np.empty(L, dtype=object)

    prior_cor = np.empty(shape=(L, T, T))
    # prior_inv = np.empty(shape=(L, T, T))
    for l in range(L):
        # prior_chol[l] = inchol(T, scale[l], inchol_tol)
        prior_cor[l, :, :] = sqexpcov(T, prior_scale[l], 1.0)
        # U, s, Vh = linalg.svd(prior_cov[l, :, :])
        # prior_inv[l, :, :] = np.dot(Vh.T, np.dot(np.diag(np.nan_to_num(np.abs(1/s))), U.T))

    # read-only variables, protection from unexpected assignment
    spike.setflags(write=0)
    prior_mean.setflags(write=0)

    # construct makeregressor
    regressor = makeregressor(spike, p, intercept)
    regressor.setflags(write=0)

    # initialize args
    # make alpha copy to avoid changing initial values
    if m0 is None:
        post_mean = prior_mean.copy()
    else:
        post_mean = m0.copy()

    if V0 is None:
        post_cov = np.empty_like(prior_cor)
        for l in range(L):
            post_cov[l, :, :] = prior_var[l] * prior_cor[l, :, :]
    else:
        post_cov = V0.copy()

    if a0 is None:
        a0 = np.random.randn(L, N)
        a0 /= linalg.norm(a0) / normofalpha
    alpha = a0.copy()

    if b0 is None:
        b0 = linalg.lstsq(regressor, spike)[0]
    beta = b0.copy()

    # initialize rate matrix, rate = E(E(spike|x))
    rate = np.empty_like(spike)
    updaterate(range(T), range(N))

    wforv = np.empty_like(post_cov)
    for l in range(L):
        wforv[l] = np.dot(rate, alpha[l, :] ** 2)

    # initialize lower bound
    lbound = np.full(maxiter + 1, np.NINF)
    lbound[0] = elbo()

    # valid values of parameters from previous iteration
    good_alpha = alpha.copy()
    good_beta = beta.copy()
    good_post_mean = post_mean.copy()
    good_post_cov = post_cov.copy()
    good_var = prior_var.copy()
    good_scale = prior_scale.copy()

    # temporary storage for recovery
    last_b = np.empty_like(beta)
    last_a = np.empty_like(alpha)
    last_m = np.empty_like(post_mean)
    last_rate = np.empty_like(rate)
    last_V = np.empty_like(post_cov)
    last_cor = np.empty_like(prior_cor)
    last_var = np.empty_like(prior_var)
    last_scale = np.empty_like(prior_scale)

    stepsize_alpha = np.ones(N)
    stepsize_beta = np.ones(N)
    stepsize_post_mean = np.ones(L)
    stepsize_scale = prior_scale.copy() * 0.1
    deflation = 0.5
    inflation = 1.5
    thld = 0.5

    # plt.figure()

    # Optimization
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it <= maxiter:
        if verbose:
            print('\nIteration[{:d}]'.format(it))
        goodLB = lbound[it - 1]
        if not fixbeta:
            for n in range(N):
                grad_b = np.dot(regressor.T, spike[:, n] - rate[:, n])
                neg_hess_b = np.dot(regressor.T, (regressor.T * rate[:, n]).T)
                if linalg.norm(grad_b, ord=np.inf) < eps:
                    break
                try:
                    delta_b = stepsize_beta[n] * linalg.solve(neg_hess_b, grad_b, sym_pos=True)
                except linalg.LinAlgError as e:
                    print('beta', e)
                    continue
                last_b[:, n] = beta[:, n]
                last_rate[:, n] = rate[:, n]
                beta[:, n] += delta_b
                updaterate(range(T), [n])
                lb = elbo()
                predicted = thld * np.inner(grad_b, delta_b)
                if np.isnan(lb) or lb < goodLB:
                    # Decrease the stepsize if the lower bound decreases.
                    # Add a small positive number to prevent becoming 0.
                    stepsize_beta[n] *= deflation
                    stepsize_beta[n] += eps
                    # Recover last valid values
                    # beta[:, n] = last_b[:, n]
                    # rate[:, n] = last_rate[:, n]
                else:
                    if lb - goodLB > predicted:
                        # Increase the stepsize if the real increment is more than expected.
                        stepsize_beta[n] *= inflation
                    goodLB = lb
            updatepostcov()

        if not fixalpha:
            for l in range(L):
                grad_a = np.dot((spike - rate).T, post_mean[:, l]) \
                         - np.dot(rate.T, post_cov[l, :, :].diagonal()) * alpha[l, :]
                neg_hess_a = np.diag(np.dot(rate.T, post_mean[:, l] ** 2)
                                     + 2 * np.dot(rate.T, post_mean[:, l] * post_cov[l, :, :].diagonal()) * alpha[l, :]
                                     + np.dot(rate.T, post_cov[l, :, :].diagonal() ** 2) * alpha[l, :] ** 2
                                     + np.dot(rate.T, post_cov[l, :, :].diagonal()))
                if linalg.norm(grad_a, ord=np.inf) < eps:
                    break
                try:
                    delta_a = stepsize_alpha[l] * linalg.solve(neg_hess_a, grad_a, sym_pos=True)
                except linalg.LinAlgError as e:
                    print('alpha', e)
                    continue
                last_a[l, :] = alpha[l, :]
                last_rate[:] = rate[:]
                alpha[l, :] += delta_a
                alpha[l, :] /= linalg.norm(alpha[l, :]) / normofalpha
                updaterate(range(T), range(N))
                lb = elbo()
                predicted = thld * np.inner(grad_a, delta_a)
                if np.isnan(lb) or lb < goodLB:
                    stepsize_alpha[l] *= deflation
                    stepsize_alpha[l] += eps
                    # alpha[l, :] = last_a[l, :]
                    # rate[:] = last_rate[:]
                else:
                    if lb - goodLB > predicted:
                        stepsize_alpha[l] *= inflation
                    goodLB = lb
            updatepostcov()

        # posterior mean
        if not fixpostmean:
            for l in range(L):
                grad_m = np.dot(spike - rate, alpha[l, :]) \
                         - np.dot(linalg.pinvh(prior_cov(l)), post_mean[:, l] - prior_mean[:, l])
                d = post_mean[:, l] - prior_mean[:, l]
                wsqrt = wroot(l)
                bmat = bmatrix(l)

                vmat = prior_cov(l) - \
                       np.dot(prior_cov(l), np.dot(wsqrt, np.dot(linalg.solve(bmat, wsqrt), prior_cov(l))))

                delta_m = stepsize_post_mean[l] * \
                          (np.dot(vmat, np.dot(spike - rate, alpha[l, :])) -
                           np.dot(eyeT -
                                  np.dot(prior_cov(l), np.dot(wsqrt, linalg.solve(bmat, wsqrt))),
                                  d))

                last_m[:, l] = post_mean[:, l]
                last_rate[:] = rate
                post_mean[:, l] += delta_m
                post_mean[:, l] -= np.mean(post_mean[:, l])
                updaterate(range(T), range(N))
                lb = elbo()
                predicted = thld * np.inner(grad_m, np.squeeze(delta_m))
                if np.isnan(lb) or lb < goodLB:
                    stepsize_post_mean[l] *= deflation
                    stepsize_post_mean[l] += eps
                    # post_mean[:, l] = last_m[:, l]
                    # rate[:] = last_rate
                else:
                    if lb - goodLB > predicted:
                        stepsize_post_mean[l] *= inflation
                    goodLB = lb
            updatepostcov()

        # posterior covariance
        # if not fixpostcov:
        #     for l in range(L):
        #         last_rate[:] = rate
        #         last_V[l, :] = post_cov[l, :]
        #         wsqrt = wroot(l)
        #         bmat = bmatrix(l)
        #         post_cov[l, :, :] = prior_cov(l) - np.dot(prior_cov(l), np.dot(wsqrt, np.dot(linalg.solve(bmat, wsqrt),
        #                                                                                      prior_cov(l))))
        #         updaterate(range(T), range(N))
        #         lb = elbo()
        #         if np.isnan(lb) or lb < goodLB:
        #             if verbose:
        #                 print('posterior covariance[{}] caused decrease'.format(l))
        #             if guardV:
        #                 rate[:] = last_rate
        #                 post_cov[l, :] = last_V[l, :]
        #         else:
        #             goodLB = lb

        if hyper and it % 5 == 0:
            for l in range(L):
                last_var[l] = prior_var[l]
                d = post_mean[:, l] - prior_mean[:, l]
                # wsqrt = wroot(l)
                # bmat = bmatrix(l)
                isv = invsigma_dot_v(l)

                candidate = (np.dot(d, np.dot(linalg.pinvh(prior_cor[l, :, :]), d)) +
                             np.trace(isv * prior_var[l])) / T

                prior_var[l] = candidate if candidate > 0 else prior_var[l] / 2

                # print('d.T * d', np.dot(d.T, d))
                # print('S inv d', np.dot(d, linalg.lstsq(cor, d)[0]))
                # print('S inv V', np.trace(linalg.lstsq(cor, post_cov[l, :, :])[0]))

                if verbose:
                    print('prior variance[{:d}]: {:.5f} -> {:.5f}'.format(l, last_var[l], prior_var[l]))
                lb = elbo()
                if np.isnan(lb) or lb < goodLB:
                    if verbose:
                        print('prior variance[{:d}] caused decrease'.format(l))
                    if guardSigma:
                        prior_var[l] = last_var[l]
                else:
                    goodLB = lb

            for _ in range(fpinter):
                for l in range(L):
                    last_scale[l] = prior_scale[l]
                    last_cor[l, :] = prior_cor[l, :]

                    d = post_mean[:, l] - prior_mean[:, l]
                    # wsqrt = wroot(l)
                    # bmat = bmatrix(l)
                    isv = invsigma_dot_v(l)
                    amat = np.dot(linalg.pinvh(prior_cor[l, :, :]), prior_cor[l, :, :] * logcor)
                    isd = np.dot(linalg.pinvh(prior_cov(l)), d)  # Sigma inverse dot d
                    grad_scale = (np.dot(isd, np.dot(prior_cov(l) * logcor, isd)) +
                                  np.trace(np.dot(amat, isv) - amat)) * prior_scale[l]
                    prior_scale[l] = np.exp(np.log(prior_scale[l]) + stepsize_scale[l] * grad_scale)
                    prior_cor[l, :, :] = sqexpcov(T, prior_scale[l], 1.0)
                    if verbose:
                        print('prior scale[{:d}]: {:.5f} -> {:.5f}'.format(l, last_scale[l], prior_scale[l]))
                    lb = elbo()
                    if np.isnan(lb) or lb < goodLB:
                        if verbose:
                            print('prior scale[{:d}] caused decrease'.format(l))
                        prior_scale[l] = last_scale[l]
                        prior_cor[l, :] = last_cor[l, :]
                        stepsize_scale[l] *= deflation
                        stepsize_scale[l] += eps
                    else:
                        if lb - goodLB > thld * np.abs(goodLB):
                            stepsize_scale[l] *= inflation
                        goodLB = lb

            updatepostcov()

        # store lower bound
        lbound[it] = elbo()

        # plt.plot(post_mean)
        # plt.draw()

        # check convergence
        chg_alpha = 0.0 if fixalpha else np.max(np.abs(good_alpha - alpha))
        chg_beta = 0.0 if fixbeta else np.max(np.abs(good_beta - beta))
        chg_post_mean = 0.0 if fixpostmean else np.max(np.abs(good_post_mean - post_mean))
        chg_post_cov = 0.0 if fixpostcov else np.max(np.abs(good_post_cov - post_cov))
        chg_variance = np.max(np.abs(good_var - prior_var)) if hyper else 0.0
        chg_scale = np.max(np.abs(good_scale - prior_scale)) if hyper else 0.0
        change = max(chg_alpha, chg_beta, chg_post_mean, chg_post_cov, chg_variance, chg_scale)

        # converged if the change in ELBO is relatively smaller than a tolerance
        if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]):
            converged = True

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'change in alpha = {:.10f}\n'
                  'change in beta = {:.10f}\n'
                  'change in posterior mean = {:.10f}\n'
                  'change in posterior covariance = {:.10f}\n'
                  'change in prior variance = {:.10f}\n'
                  'change in prior scale = {:.10f}'.format(lbound[it], lbound[it] - lbound[it - 1],
                                                           chg_alpha, chg_beta, chg_post_mean,
                                                           chg_post_cov, chg_variance, chg_scale))

        # store current iteration
        good_alpha[:] = alpha
        good_beta[:] = beta
        good_post_mean[:] = post_mean
        good_post_cov[:] = post_cov
        good_var[:] = prior_var
        good_scale[:] = prior_scale

        it += 1

    stop = timeit.default_timer()

    return lbound[:it], post_mean, post_cov, alpha, beta, prior_var, prior_scale, a0, b0, stop - start, converged
