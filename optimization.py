import itertools
import timeit
import numpy as np
from scipy import linalg


# from numpy import linalg
from util import makeregressor, inchol, sqexpcov


def likelihood(spike, latent, alpha, beta, intercept=True):
    T, N = spike.shape
    L, _ = latent.shape
    k, _ = beta.shape
    p = (k - intercept) // N

    regressor = makeregressor(spike, p, intercept)

    lograte = np.dot(regressor, beta) + np.dot(latent, alpha)
    return np.sum(spike * lograte - np.exp(lograte))


def saferate(t, n, regressor, post_mean, post_cov, beta, alpha):
    lograte = np.dot(regressor[t, :], beta[:, n]) + np.dot(post_mean[t, :], alpha[:, n]) \
              + 0.5 * np.sum(alpha[:, n] ** 2 * post_cov[:, t, t])
    rate = np.nan_to_num(np.exp(lograte))
    return rate if rate > 0 else np.finfo(np.float).eps


def lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov, regressor=None, rate=None):
    """
    Calculate the lower bound
    :param prior_var:
    :param spike: (T, N), spike trains
    :param beta: (1 + p*N, N), coefficients of spike
    :param alpha: (L, N), coefficients of x
    :param prior_mean: (T, L), prior mean
    :param post_mean: (T, L), latent posterior mean
    :param post_cov: (L, T, T), latent posterior covariances
    :param complete: compute constant terms
    :param regressor: (T, 1 + p*N), vectorized spike makeregressor
    :param rate: (T, N), E(E(spike|x))
    :return lbound: lower bound
    """

    _, L = prior_mean.shape
    T, N = spike.shape

    lbound = np.sum(spike * (np.dot(regressor, beta) + np.dot(post_mean, alpha)) - rate)

    for l in range(L):
        lbound += -0.5 * np.dot(post_mean[:, l] - prior_mean[:, l],
                                linalg.lstsq(prior_cor[l, :, :] * prior_var[l],
                                             post_mean[:, l] - prior_mean[:, l])[0]) - \
                  0.5 * np.trace(linalg.lstsq(prior_cor[l, :, :] * prior_var[l], post_cov[l, :, :])[0]) - \
                  0.5 * np.linalg.slogdet(linalg.lstsq(prior_cor[l, :, :] * prior_var[l], post_cov[l, :, :])[0])[1]
                  # 0.5 * np.linalg.slogdet(post_cov[l, :, :])[1] - \
                  # 0.5 * np.linalg.slogdet(prior_cor[l, :, :] * prior_var[l])[1]

    return lbound


def variational(spike, p, prior_mean, var, scale,
                a0=None, b0=None, m0=None, V0=None,
                guardV=True, guardSigma=True,
                fixalpha=False, fixbeta=False, fixpostmean=False, fixpostcov=False, normofalpha=1.0, intercept=True,
                hyper=False, inchol_tol=1e-7,
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
    :param K0: (L, T, T), initial value of posterior covariance inverse
    :param fixalpha: bool, switch of not optimize alpha
    :param fixbeta: bool, switch of not optimize beta
    :param fixpostmean: bool, switch of not optimize posterior mean
    :param fixpostcov: bool, switch of not optimize posterior covariance
    :param normofalpha: norm constraint of alpha
    :param intercept: bool, include intercept term or not
    :param hyper: optimize hyperparameters or not
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

    start = timeit.default_timer()  # time when algorithm starts

    def updaterate(rows, cols):
        # rate = E(E(spike|x))
        for row, col in itertools.product(rows, cols):
            rate[row, col] = saferate(row, col, regressor, post_mean, post_cov, beta, alpha)

    # control
    maxiter = control['max iteration']
    fpinter = control['fixed-point iteration']
    tol = control['tol']
    verbose = control['verbose']

    # epsilon
    eps = 2 * np.finfo(np.float).eps

    # dimensions
    T, N = spike.shape
    _, L = prior_mean.shape

    eyeT = np.identity(T)

    # hyperparameters
    prior_var = var.copy()
    prior_scale = scale.copy()

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

    # initialize lower bound
    lbound = np.full(maxiter + 1, np.NINF)
    lbound[0] = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                           regressor=regressor, rate=rate)

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
    stepsize_scale = scale.copy() * 0.1
    deflation = 0.5
    inflation = 1.5
    thld = 0.1

    # plt.figure()

    # Optimization
    it = 1
    converged = False
    while converged < 2 and it <= maxiter:
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
                    delta_b = stepsize_beta[n] * linalg.solve(neg_hess_b, grad_b)
                except linalg.LinAlgError as e:
                    print('beta', e)
                    continue
                last_b[:, n] = beta[:, n]
                last_rate[:, n] = rate[:, n]
                beta[:, n] += delta_b
                updaterate(range(T), [n])
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                                regressor=regressor, rate=rate)
                if np.isnan(lb) or lb < goodLB:
                    # Decrease the stepsize if the lower bound decreases.
                    # Add a small positive number to prevent becoming 0.
                    stepsize_beta[n] *= deflation
                    stepsize_beta[n] += eps
                    # Recover last valid values
                    beta[:, n] = last_b[:, n]
                    rate[:, n] = last_rate[:, n]
                else:
                    if lb - goodLB > thld * goodLB:
                        # Increase the stepsize if the real increment is more than expected.
                        stepsize_beta[n] *= inflation
                        # if stepsize_beta[n] > 1:
                        #     stepsize_beta[n] = 1.0
                    goodLB = lb

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
                    delta_a = stepsize_alpha[l] * linalg.solve(neg_hess_a, grad_a)
                except linalg.LinAlgError as e:
                    print('alpha', e)
                    continue
                last_a[l, :] = alpha[l, :]
                last_rate[:] = rate[:]
                alpha[l, :] += delta_a
                alpha[l, :] /= linalg.norm(alpha[l, :]) / normofalpha
                updaterate(range(T), range(N))
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                                regressor=regressor, rate=rate)
                if np.isnan(lb) or lb < goodLB:
                    stepsize_alpha[l] *= deflation
                    stepsize_alpha[l] += eps
                    alpha[l, :] = last_a[l, :]
                    rate[:] = last_rate[:]
                else:
                    if lb - goodLB > thld * goodLB:
                        stepsize_alpha[l] *= inflation
                        # if stepsize_alpha[l] > 1:
                        #     stepsize_alpha[l] = 1.0
                    goodLB = lb

        # posterior mean
        if not fixpostmean:
            for l in range(L):
                # grad_m = np.dot(spike - rate, alpha[l, :]) \
                #          - linalg.lstsq(prior_cor[l, :, :] * prior_var[l], post_mean[:, l] - prior_mean[:, l])[0]
                w = np.dot(rate, alpha[l, :] ** 2)
                wsqrt = np.mat(np.diag(np.sqrt(w)))
                d = np.mat(post_mean[:, l] - prior_mean[:, l]).T
                bmat = eyeT + wsqrt * np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt
                # bchol = linalg.cholesky(bmat, lower=True)
                # hinv = prior_cor[l, :, :] * prior_var[l] - \
                #        np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt * \
                #        np.mat(linalg.lstsq(bchol.T,
                #                            linalg.lstsq(bchol,
                #                                         wsqrt * np.mat(prior_cor[l, :, :] * prior_var[l]))[0])[0])
                # delta_m = stepsize_post_mean[l] * (np.mat(hinv) * np.mat(np.dot(spike - rate, alpha[l, :])).T - d +
                #                                    np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt *
                #                                    np.mat(linalg.lstsq(bchol.T, linalg.lstsq(bchol, wsqrt * d)[0])[0]))

                hinv = prior_cor[l, :, :] * prior_var[l] - \
                       np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt * \
                       np.mat(linalg.lstsq(bmat, wsqrt * np.mat(prior_cor[l, :, :] * prior_var[l]))[0])
                delta_m = stepsize_post_mean[l] * (np.mat(hinv) * np.mat(np.dot(spike - rate, alpha[l, :])).T - d +
                                                   np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt *
                                                   np.mat(linalg.lstsq(bmat, wsqrt * d)[0]))

                last_m[:, l] = post_mean[:, l]
                last_rate[:] = rate
                post_mean[:, l] += delta_m.flat
                post_mean[:, l] -= np.mean(post_mean[:, l])
                updaterate(range(T), range(N))
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                                regressor=regressor, rate=rate)
                if np.isnan(lb) or lb < goodLB:
                    stepsize_post_mean[l] *= deflation
                    stepsize_post_mean[l] += eps
                    post_mean[:, l] = last_m[:, l]
                    rate[:] = last_rate
                else:
                    if lb - goodLB > thld * goodLB:
                        stepsize_post_mean[l] *= inflation
                        # if stepsize_post_mean[l] > 1:
                        #     stepsize_post_mean[l] = 1.0
                    goodLB = lb

        # posterior covariance
        if not fixpostcov:
            for l in range(L):
                last_rate[:] = rate
                last_V[l, :] = post_cov[l, :]
                w = np.dot(rate, alpha[l, :] ** 2)
                wsqrt = np.mat(np.diag(np.sqrt(w)))
                bmat = eyeT + wsqrt * np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt
                # bchol = linalg.cholesky(bmat, lower=True)
                # post_cov[l, :, :] = prior_cor[l, :, :] * prior_var[l] - \
                #                     np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt * \
                #                     np.mat(linalg.lstsq(bchol.T,
                #                                         linalg.lstsq(bchol,
                #                                                      wsqrt * np.mat(prior_cor[l, :, :] *
                #                                                                     prior_var[l]))[0])[0])
                post_cov[l, :, :] = prior_cor[l, :, :] * prior_var[l] - \
                                    np.mat(prior_cor[l, :, :] * prior_var[l]) * wsqrt * \
                                    np.mat(linalg.lstsq(bmat, wsqrt * np.mat(prior_cor[l, :, :] * prior_var[l]))[0])
                updaterate(range(T), range(N))
                if guardV:
                    lb = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                                    regressor=regressor, rate=rate)
                    if np.isnan(lb) or lb < goodLB:
                        if verbose:
                            print('posterior covariance[{}] caused decrease'.format(l))
                        rate[:] = last_rate
                        post_cov[l, :] = last_V[l, :]
                    else:
                        goodLB = lb

        if hyper and it % 5 == 0:
            for l in range(L):
                last_var[l] = prior_var[l]

                d = post_mean[:, l] - prior_mean[:, l]
                candidate = (np.dot(d, linalg.lstsq(prior_cor[l, :, :], d)[0]) +
                             np.trace(linalg.lstsq(prior_cor[l, :, :], post_cov[l, :, :])[0])) / T
                if candidate < 0:
                    print(linalg.eigvals(prior_cor[l, :, :]))
                    w, v = linalg.eigh(prior_cor[l, :, :])
                    invw = np.zeros_like(w)
                    invw[np.ma.masked_greater(w, 10 * eps).mask] = 1 / w[np.ma.masked_greater(w, 10 * eps).mask]
                    print(invw)
                    invs = np.dot(v, np.dot(np.diag(invw), v))
                    print('S inverse * m', np.dot(d, np.dot(invs, d)))
                    print('trace of S inverse * V', np.trace(np.dot(invs, post_cov[l, :, :])))

                prior_var[l] = candidate if candidate > 0 else prior_var[l] / 2

                # print('d.T * d', np.dot(d.T, d))
                # print('S inv d', np.dot(d, linalg.lstsq(cor, d)[0]))
                # print('S inv V', np.trace(linalg.lstsq(cor, post_cov[l, :, :])[0]))

                if verbose:
                    print('prior variance[{:d}] -> {:.5f}'.format(l, prior_var[l]))
                if guardSigma:
                    lb = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                                    regressor=regressor, rate=rate)
                    if np.isnan(lb) or lb < goodLB:
                        if verbose:
                            print('prior variance[{:d}] caused decrease'.format(l))
                        prior_var[l] = last_var[l]
                    else:
                        goodLB = lb

            for l in range(L):
                last_scale[l] = prior_scale[l]
                last_cor[l, :] = prior_cor[l, :]

                d = post_mean[:, l] - prior_mean[:, l]
                amat = linalg.lstsq(prior_cor[l, :, :], prior_cor[l, :, :] * logcor)[0]
                grad_scale = (np.dot(d, np.dot(amat, linalg.lstsq(prior_cor[l, :, :] * prior_var[l], d)[0])) +
                              np.trace(np.dot(amat,
                                              linalg.lstsq(prior_cor[l, :, :] * prior_var[l],
                                                           post_cov[l, :, :])[0]) - amat)) * prior_scale[l]
                prior_scale[l] = np.exp(np.log(prior_scale[l]) + stepsize_scale[l] * grad_scale)
                prior_cor[l, :, :] = sqexpcov(T, prior_scale[l], 1.0)
                if verbose:
                    print('prior scale[{:d}] -> {:.5f}'.format(l, prior_scale[l]))
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                                regressor=regressor, rate=rate)
                if np.isnan(lb) or lb < goodLB:
                    if verbose:
                        print('prior scale[{:d}] caused decrease'.format(l))
                    prior_scale[l] = last_scale[l]
                    prior_cor[l, :] = last_cor[l, :]
                    stepsize_scale[l] *= deflation
                    stepsize_scale[l] += eps
                else:
                    if lb - goodLB > thld * goodLB:
                        stepsize_scale[l] *= inflation
                    goodLB = lb

        # store lower bound
        lbound[it] = lowerbound(spike, beta, alpha, prior_mean, prior_var, prior_cor, post_mean, post_cov,
                                regressor=regressor, rate=rate)

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

        if change < tol:
            converged += 1
        else:
            converged = 0

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
        good_scale[:] = scale

        it += 1

    stop = timeit.default_timer()

    return lbound[:it], post_mean, post_cov, alpha, beta, prior_var, prior_scale, a0, b0, stop - start, converged
