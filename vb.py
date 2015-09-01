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


def lowerbound(spike, beta, alpha, prior_mean, prior_cov, prior_inv, post_mean, post_cov, regressor=None, rate=None):
    """
    Calculate the lower bound
    :param spike: (T, N), spike trains
    :param beta: (1 + p*N, N), coefficients of spike
    :param alpha: (L, N), coefficients of x
    :param prior_mean: (T, L), prior mean
    :param prior_inv: (L, T, T), prior inverse covariances
    :param post_mean: (T, L), latent posterior mean
    :param post_cov: (L, T, T), latent posterior covariances
    :param complete: compute constant terms
    :param regressor: (T, 1 + p*N), vectorized spike makeregressor
    :param rate: (T, N), E(E(spike|x))
    :return lbound: lower bound
    """

    eps = 2 * np.finfo(np.float).eps
    _, L = prior_mean.shape
    T, N = spike.shape
    eyeT = np.identity(T)

    lbound = np.sum(spike * (np.dot(regressor, beta) + np.dot(post_mean, alpha)) - rate)

    for l in range(L):
        # lbound += -0.5 * np.dot(post_mean[:, l] - prior_mean[:, l],
        #                         np.dot(prior_inv[l, :, :], post_mean[:, l] - prior_mean[:, l])) \
        #           - 0.5 * np.trace(np.dot(prior_inv[l, :, :], post_cov[l, :, :])) \
        #           + 0.5 * np.linalg.slogdet(post_cov[l, :, :])[1]
        lbound += -0.5 * np.dot(post_mean[:, l] - prior_mean[:, l],
                                linalg.solve(prior_cov[l, :, :], post_mean[:, l] - prior_mean[:, l])) \
                  - 0.5 * np.trace(linalg.solve(prior_cov[l, :, :], post_cov[l, :, :])) \
                  + 0.5 * np.linalg.slogdet(post_cov[l, :, :])[1]

    return lbound

default_control = {'maxiter': 200,
                   'fixed-point iteration': 3,
                   'tol': 1e-4,
                   'verbose': False}


def variational(spike, p, prior_mean, prior_var, prior_w,
                a0=None, b0=None, m0=None,
                fixalpha=False, fixbeta=False, fixpostmean=False, fixpostcov=False, normofalpha=1.0, intercept=True,
                hyper=False, inchol_tol=1e-7,
                control=default_control):
    """
    :param spike: (T, N), spike trains
    :param p: order of regression
    :param prior_mean: (T, L), prior mean
    :param prior_var: (L,), prior variance
    :param prior_w: (L,), prior inverse of squared lengthscale
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
    maxiter = control['maxiter']
    fpinter = control['fixed-point iteration']
    tol = control['tol']
    verbose = control['verbose']

    # epsilon
    eps = 2 * np.finfo(np.float).eps

    # dimensions
    T, N = spike.shape
    _, L = prior_mean.shape

    eyeT = np.identity(T)

    variance = prior_var.copy()
    w = prior_w.copy()

    prior_chol = np.empty(L, dtype=object)
    prior_cov = np.empty(shape=(L, T, T))
    prior_inv = np.empty(shape=(L, T, T))
    for l in range(L):
        prior_chol[l] = inchol(T, w[l], inchol_tol)
        prior_cov[l, :, :] = sqexpcov(T, w[l], variance[l]) + eyeT * 1e-7
        # pinv = linalg.lstsq(prior_chol[l], eyeT)[0]
        # prior_inv[l, :, :] = np.dot(pinv.T, pinv) / variance[l]
        prior_inv[l, :, :] = linalg.inv(prior_cov[l, :, :])

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

    post_cov = prior_cov.copy()
    post_inv = prior_inv.copy()

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
    lbound = np.full(maxiter, np.NINF)
    lbound[0] = lowerbound(spike, beta, alpha, prior_mean, prior_cov, prior_inv, post_mean, post_cov, regressor=regressor, rate=rate)

    # valid values of parameters from previous iteration
    good_alpha = alpha.copy()
    good_beta = beta.copy()
    good_post_mean = post_mean.copy()
    good_post_cov = post_cov.copy()

    # temporary storage for recovery
    last_b = beta.copy()
    last_a = alpha.copy()
    last_m = post_mean.copy()
    last_rate = rate.copy()
    last_V = np.empty((T, T))

    stepsize_alpha = np.ones(N)
    stepsize_beta = np.ones(N)
    stepsize_post_mean = np.ones(L)
    deflation = 0.5
    inflation = 1.5
    thld = 0.75

    # Optimization
    it = 1
    converged = False
    while not converged and it < maxiter:
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
                predict = np.inner(grad_b, delta_b) - 0.5 * np.dot(delta_b, np.dot(neg_hess_b, delta_b))
                beta[:, n] += delta_b
                updaterate(range(T), [n])
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_cov, prior_inv, post_mean, post_cov, regressor=regressor, rate=rate)
                if np.isnan(lb) or lb < lbound[it - 1]:
                    # Decrease the stepsize if the lower bound decreases.
                    # Add a small positive number to prevent becoming 0.
                    stepsize_beta[n] *= deflation
                    stepsize_beta[n] += eps
                    # Recover last valid values
                    beta[:, n] = last_b[:, n]
                    rate[:, n] = last_rate[:, n]
                elif lb - lbound[it - 1] > thld * predict:
                    # Increase the stepsize if the real increment is more than expected.
                    stepsize_beta[n] *= inflation
                    # if stepsize_beta[n] > 1:
                    #     stepsize_beta[n] = 1.0

        if not fixalpha:
            for l in range(L):
                grad_a = np.dot((spike - rate).T, post_mean[:, l]) - np.dot(rate.T, post_cov[l, :, :].diagonal()) * alpha[l, :]
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
                predict = np.inner(grad_a, delta_a) - 0.5 * np.dot(delta_a, np.dot(neg_hess_a, delta_a))
                updaterate(range(T), range(N))
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_cov, prior_inv, post_mean, post_cov, regressor=regressor, rate=rate)
                if np.isnan(lb) or lb - lbound[it - 1] < 0:
                    stepsize_alpha[l] *= deflation
                    stepsize_alpha[l] += eps
                    alpha[l, :] = last_a[l, :]
                    rate[:] = last_rate[:]
                elif lb - lbound[it - 1] > thld * predict:
                    stepsize_alpha[l] *= inflation
                    # if stepsize_alpha[l] > 1:
                    #     stepsize_alpha[l] = 1.0
                # Scale norm
                alpha[l, :] /= linalg.norm(alpha[l, :]) / normofalpha

        # posterior mean
        if not fixpostmean:
            for l in range(L):
                grad_m = np.dot(spike - rate, alpha[l, :]) \
                         - linalg.solve(prior_cov[l, :, :], post_mean[:, l] - prior_mean[:, l])
                neg_hess_m = np.diag(np.dot(rate, alpha[l, :] ** 2)) + prior_inv[l, :, :]
                if linalg.norm(grad_m, ord=np.inf) < eps:
                    break
                try:
                    delta_m = stepsize_post_mean[l] * linalg.solve(neg_hess_m, grad_m)
                except linalg.LinAlgError as e:
                    print('post_mean', e)
                    continue
                last_m[:, l] = post_mean[:, l]
                last_rate[:] = rate
                post_mean[:, l] += delta_m
                predict = np.inner(grad_m, delta_m) - 0.5 * np.dot(delta_m, np.dot(neg_hess_m, delta_m))
                updaterate(range(T), range(N))
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_cov, prior_inv, post_mean, post_cov, regressor=regressor, rate=rate)
                if np.isnan(lb) or lb < lbound[it - 1]:
                    stepsize_post_mean[l] *= deflation
                    stepsize_post_mean[l] += eps
                    post_mean[:, l] = last_m[:, l]
                    rate[:] = last_rate
                elif lb - lbound[it - 1] > thld * predict:
                    stepsize_post_mean[l] *= inflation
                    # if stepsize_post_mean[l] > 1:
                    #     stepsize_post_mean[l] = 1.0
                # Shift location
                post_mean[:, l] -= np.mean(post_mean[:, l])

        # posterior covariance
        if not fixpostcov:
            for l in range(L):
                for t in range(T):
                    last_rate[t, :] = rate[t, :]
                    last_V[:] = post_cov[l, :, :]
                    k_ = post_inv[l, t, t] - 1 / post_cov[l, t, t]  # \tilde{k}_tt
                    old_vtt = post_cov[l, t, t]
                    # fixed point iterations
                    for _ in range(fpinter):
                        post_cov[l, t, t] = 1 / (prior_inv[l, t, t] - k_ + np.sum(rate[t, :] * alpha[l, :] ** 2))
                        updaterate([t], range(N))
                    # update post_cov
                    not_t = np.arange(T) != t
                    post_cov[np.ix_([l], not_t, not_t)] = post_cov[np.ix_([l], not_t, not_t)] \
                                                   + (post_cov[l, t, t] - old_vtt) \
                                                   * np.outer(post_cov[l, t, not_t], post_cov[l, t, not_t]) / (old_vtt * old_vtt)
                    post_cov[l, t, not_t] = post_cov[l, not_t, t] = post_cov[l, t, t] * post_cov[l, t, not_t] / old_vtt
                    # update k_tt
                    post_inv[l, t, t] = k_ + 1 / post_cov[l, t, t]
                    # lb = lowerbound(spike, beta, alpha, prior_mean, prior_cov, prior_inv, post_mean, post_cov, regressor=regressor, rate=rate)
                    # if np.isnan(lb) or lb < lbound[it - 1]:
                    #     print('post_cov[{}] decreased L.'.format(l))
                        # post_inv[l, t, t] = k_ + 1 / post_cov[l, t, t]
                        # post_cov[l, :, :] = last_V
                        # rate[t, :] = last_rate[t, :]

        if hyper:
            for l in range(L):
                vl = np.trace(np.dot(prior_inv[l, :, :] * variance[l],
                                     np.outer(post_mean - prior_mean, post_mean - prior_mean)
                                     + post_cov[l, :, :])) / T
                prior_cov[l, :, :] *= vl / variance[l]
                prior_inv[l, :, :] /= vl / variance[l]
                variance[l] = vl
                print(vl)

        # update lower bound
        lbound[it] = lowerbound(spike, beta, alpha, prior_mean, prior_cov, prior_inv, post_mean, post_cov,
                                regressor=regressor, rate=rate)

        # check convergence
        chg_alpha = 0.0 if fixalpha else np.max(np.abs(good_alpha - alpha))
        chg_beta = 0.0 if fixbeta else np.max(np.abs(good_beta - beta))
        chg_post_mean = 0.0 if fixpostmean else np.max(np.abs(good_post_mean - post_mean))
        chg_post_cov = 0.0 if fixpostcov else np.max(np.abs(good_post_cov - post_cov))
        change = max(chg_alpha, chg_beta, chg_post_mean, chg_post_cov)

        if change < tol:
            converged = True

        if verbose:
            print('\nIteration[%d]: L = %.5f, inflation = %.10f' %
                  (it + 1, lbound[it], lbound[it] - lbound[it - 1]))
            print('change in alpha = %.10f' % chg_alpha)
            print('change in beta = %.10f' % chg_beta)
            print('change in posterior mean = %.10f' % chg_post_mean)
            print('change in posterior covariance = %.10f' % chg_post_cov)

        good_alpha[:] = alpha
        good_beta[:] = beta
        good_post_mean[:] = post_mean
        good_post_cov[:] = post_cov

        it += 1

    stop = timeit.default_timer()

    return lbound[:it], post_mean, post_cov, alpha, beta, a0, b0, stop - start, converged
