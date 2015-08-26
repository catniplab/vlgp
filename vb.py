import itertools
import warnings
import time
import numpy as np
from util import history


def likelihood(spike, latent, a, b, intercept=True):
    T, N = spike.shape
    L, _ = latent.shape
    k, _ = b.shape
    p = (k - intercept) // N

    regressor = history(spike, p, intercept)

    lograte = np.dot(regressor, b) + np.dot(latent, a)
    return np.sum(spike * lograte - np.exp(lograte))


def saferate(t, n, regressor, post_mean, post_cov, beta, alpha):
    lograte = np.dot(regressor[t, :], beta[:, n]) + np.dot(post_mean[t, :], alpha[:, n]) + 0.5 * np.sum(alpha[:, n] * alpha[:, n] * post_cov[:, t, t])
    rate = np.nan_to_num(np.exp(lograte))
    return rate if rate > 0 else np.finfo(np.float).eps


def lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, complete=False, regressor=None, frate=None):
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
    :param regressor: (T, 1 + p*N), vectorized spike history
    :param frate: (T, N), E(E(spike|x))
    :return lbound: lower bound
    """

    _, L = prior_mean.shape
    T, N = spike.shape

    lbound = np.sum(spike * (np.dot(regressor, beta) + np.dot(post_mean, alpha)) - frate)

    for l in range(L):
        lbound += -0.5 * np.dot(post_mean[:, l] - prior_mean[:, l], np.dot(prior_inv[l, :, :], post_mean[:, l] - prior_mean[:, l])) + \
                  -0.5 * np.trace(np.dot(prior_inv[l, :, :], post_cov[l, :, :])) + 0.5 * np.linalg.slogdet(post_cov[l, :, :])[1]

    return lbound + 0.5 * np.sum(np.linalg.slogdet(prior_inv)[1]) if complete else lbound

default_control = {'maxiter': 200,
                   'fixed-point iteration': 3,
                   'tol': 1e-4,
                   'verbose': False}


def variational(spike, p, prior_mean, prior_cov, prior_inv=None,
                a0=None, b0=None, m0=None, V0=None, K0=None,
                fixa=False, fixb=False, fixm=False, fixV=False, anorm=1.0, intercept=True,
                constrain_m='lag', constrain_a='lag',
                control=default_control):
    """
    :param spike: (T, N), spike trains
    :param prior_mean: (T, L), prior mean
    :param prior_cov: (L, T, T), prior covariance
    :param prior_inv: (L, T, T), inverse prior covariance
    :param p: order of autoregression
    :param maxiter: maximum number of iterations
    :param tol: convergence tolerance
    :return
        post_mean: posterior mean
        post_cov: posterior covariance
        beta: coefficients of spike
        alpha: coefficients of x
        lbound: lower bound sequence
        it: number of iterations
    """
    start = time.time()  # time when algorithm starts

    def updaterate(t, n):
        # rate = E(E(spike|x))
        for t, n in itertools.product(t, n):
            rate[t, n] = saferate(t, n, regressor, post_mean, post_cov, beta, alpha)

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

    eyeL = np.identity(L)
    eyeN = np.identity(N)
    eyeT = np.identity(T)
    oneT = np.ones(T)
    jayT = np.ones((T, T))
    oneTL = np.ones((T, L))

    # calculate inverse of prior covariance if not given
    if prior_inv is None:
        prior_inv = np.empty_like(prior_cov)
        for l in range(L):
            prior_inv[l, :, :] = np.linalg.inv(prior_cov[l, :, :])

    # read-only variables, protection from unexpected assignment
    spike.setflags(write=0)
    prior_mean.setflags(write=0)
    # prior_cov.setflags(write=0)
    # prior_inv.setflags(write=0)

    # construct history
    regressor = history(spike, p, intercept)

    # initialize args
    # make alpha copy to avoid changing initial values
    if m0 is None:
        post_mean = prior_mean.copy()
    else:
        post_mean = m0.copy()

    if V0 is None:
        post_cov = prior_cov.copy()
    else:
        post_cov = V0.copy()

    if K0 is None:
        post_inv = prior_inv.copy()
    else:
        post_inv = np.empty_like(post_cov)
        for l in range(L):
            post_inv[l, :, :] = np.linalg.inv(post_cov[l, :, :])

    if a0 is None:
        a0 = np.random.randn(L, N)
        a0 /= np.linalg.norm(a0) / anorm
    alpha = a0.copy()

    if b0 is None:
        b0 = np.linalg.lstsq(regressor, spike)[0]
    beta = b0.copy()

    # initialize rate matrix, rate = E(E(spike|x))
    rate = np.empty_like(spike)
    updaterate(range(T), range(N))

    # initialize lower bound
    lbound = np.full(maxiter, np.NINF)
    lbound[0] = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, frate=rate)

    # old values
    old_a = alpha.copy()
    old_b = beta.copy()
    old_m = post_mean.copy()
    old_V = post_cov.copy()

    # variables for recovery
    last_b = beta.copy()
    last_a = alpha.copy()
    last_m = post_mean.copy()
    last_rate = rate.copy()
    last_V = np.empty((T, T))

    ra = np.ones(N)
    rb = np.ones(N)
    rm = np.ones(L)
    dec = 0.5
    inc = 1.5
    thld = 0.75

    # gradient and hessian
    grad_a_lag = np.zeros(N + 1)
    hess_a_lag = np.zeros((grad_a_lag.size, grad_a_lag.size))
    lam_a = np.zeros(L)
    lam_last_a = lam_a.copy()

    grad_m_lag = np.zeros(T + 1)
    hess_m_lag = np.zeros((grad_m_lag.size, grad_m_lag.size))
    lam_m = np.zeros(L)
    lam_last_m = lam_m.copy()

    it = 1
    convergent = False
    while not convergent and it < maxiter:
        if not fixb:
            for n in range(N):
                grad_b = np.dot(regressor.T, spike[:, n] - rate[:, n])
                hess_b = np.dot(regressor.T, (regressor.T * -rate[:, n]).T)
                if np.linalg.norm(grad_b, ord=np.inf) < eps:
                    break
                try:
                    delta_b = -rb[n] * np.linalg.solve(hess_b, grad_b)
                except np.linalg.LinAlgError as e:
                    print('beta', e)
                    continue
                last_b[:, n] = beta[:, n]
                last_rate[:, n] = rate[:, n]
                predict = np.inner(grad_b, delta_b) + 0.5 * np.dot(delta_b, np.dot(hess_b, delta_b))
                beta[:, n] += delta_b
                updaterate(range(T), [n])
                lb = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, frate=rate)
                if np.isnan(lb) or lb < lbound[it - 1]:
                    rb[n] = dec * rb[n] + eps
                    beta[:, n] = last_b[:, n]
                    rate[:, n] = last_rate[:, n]
                elif lb - lbound[it - 1] > thld * predict:
                    rb[n] *= inc
                    # if rb[n] > 1:
                    #     rb[n] = 1.0

        if not fixa:
            for l in range(L):
                grad_a = np.dot((spike - rate).T, post_mean[:, l]) - np.dot(rate.T, post_cov[l, :, :].diagonal()) * alpha[l, :]
                hess_a = -np.diag(np.dot(rate.T, post_mean[:, l] * post_mean[:, l])
                                  + 2 * np.dot(rate.T, post_mean[:, l] * post_cov[l, :, :].diagonal()) * alpha[l, :]
                                  + np.dot(rate.T, post_cov[l, :, :].diagonal() ** 2) * alpha[l, :] ** 2
                                  + np.dot(rate.T, post_cov[l, :, :].diagonal()))
                if constrain_a == 'lag':
                    grad_a_lag[:N] = grad_a + 2 * lam_a[l] * alpha[l, :]
                    grad_a_lag[N:] = np.inner(alpha[l, :], alpha[l, :]) - anorm ** 2
                    hess_a_lag[:N, :N] = hess_a + 2 * lam_a[l] * eyeN
                    hess_a_lag[N:, :N] = 2 * alpha[l, :]
                    hess_a_lag[:N, N:] = hess_a_lag[N:, :N].T
                    hess_a_lag[N:, N:] = 0
                    if np.linalg.norm(grad_a_lag, ord=np.inf) < eps:
                        break
                    try:
                        delta_a_lag = -ra[l] * np.linalg.solve(hess_a_lag, grad_a_lag)
                    except np.linalg.LinAlgError as e:
                        print('alpha', e)
                        continue
                    lam_last_a[l] = lam_a[l]
                    last_a[l, :] = alpha[l, :]
                    last_rate[:] = rate[:]
                    predict = np.inner(grad_a_lag, delta_a_lag) \
                              + 0.5 * np.dot(delta_a_lag, np.dot(hess_a_lag, delta_a_lag))
                    alpha[l, :] += delta_a_lag[:N]
                    lam_a[l] += delta_a_lag[N:]
                    updaterate(range(T), range(N))
                    lb = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, frate=rate)
                    if np.isnan(lb) or lb - lbound[it - 1] < 0:
                        ra[l] *= dec
                        ra[l] += eps
                        alpha[l, :] = last_a[l, :]
                        rate[:] = last_rate[:]
                        lam_a[l] = lam_last_a[l]
                    elif lb - lbound[it - 1] > thld * predict:
                        ra[l] *= inc
                        # if ra[l] > 1:
                        #     ra[l] = 1.0
                else:
                    if np.linalg.norm(grad_a, ord=np.inf) < eps:
                        break
                    try:
                        delta_a = -ra[l] * np.linalg.solve(hess_a, grad_a)
                    except np.linalg.LinAlgError as e:
                        print('alpha', e)
                        continue
                    last_a[l, :] = alpha[l, :]
                    last_rate[:] = rate[:]
                    alpha[l, :] += delta_a
                    if np.linalg.norm(alpha[l, :]) > 0:
                        predict = np.inner(grad_a, delta_a / np.linalg.norm(alpha[l, :]) * anorm) \
                                  + 0.5 * np.dot(delta_a / np.linalg.norm(alpha[l, :]) * anorm,
                                                 np.dot(hess_a, delta_a / np.linalg.norm(alpha[l, :]) * anorm))
                        alpha[l, :] /= np.linalg.norm(alpha[l, :]) / anorm
                    else:
                        predict = np.inner(grad_a, delta_a) + 0.5 * np.dot(delta_a, np.dot(hess_a, delta_a))

                    updaterate(range(T), range(N))
                    lb = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, frate=rate)
                    if np.isnan(lb) or lb - lbound[it - 1] < 0:
                        ra[l] *= dec
                        ra[l] += eps
                        alpha[l, :] = last_a[l, :]
                        rate[:] = last_rate[:]
                    elif lb - lbound[it - 1] > thld * predict:
                        ra[l] *= inc
                        # if ra[l] > 1:
                        #     ra[l] = 1.0

        # posterior mean
        if not fixm:
            for l in range(L):
                grad_m = np.dot(spike - rate, alpha[l, :]) - np.dot(prior_inv[l, :, :], post_mean[:, l] - prior_mean[:, l])
                hess_m = np.diag(np.dot(-rate, alpha[l, :] * alpha[l, :])) - prior_inv[l, :, :]
                if constrain_m == 'lag':
                    grad_m_lag[:T] = grad_m + lam_m[l]
                    grad_m_lag[T:] = np.sum(post_mean[:, l])
                    hess_m_lag[:T, :T] = hess_m
                    hess_m_lag[:T, T:] = hess_m_lag[T:, :T] = 1
                    hess_m_lag[T:, T:] = 0

                    if np.linalg.norm(grad_m_lag, ord=np.inf) < eps:
                        break
                    try:
                        delta_m_lag = -rm[l] * np.linalg.solve(hess_m_lag, grad_m_lag)
                    except np.linalg.LinAlgError as e:
                        print('post_mean', e)
                        continue
                    last_m[:, l] = post_mean[:, l]
                    last_rate[:] = rate
                    lam_last_m[l] = lam_m[l]
                    predict = np.inner(grad_m_lag, delta_m_lag) + 0.5 * np.dot(delta_m_lag, np.dot(hess_m_lag, delta_m_lag))
                    post_mean[:, l] += delta_m_lag[:T]
                    lam_m[l] += delta_m_lag[T:]
                    updaterate(range(T), range(N))
                    lb = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, frate=rate)
                    if np.isnan(lb) or lb < lbound[it - 1]:
                        rm[l] *= dec
                        rm[l] += eps
                        post_mean[:, l] = last_m[:, l]
                        rate[:] = last_rate
                        lam_m[l] = lam_last_m[l]
                    elif lb - lbound[it - 1] > thld * predict:
                        rm[l] *= inc
                        # if rm[l] > 1:
                        #     rm[l] = 1.0
                else:
                    if np.linalg.norm(grad_m, ord=np.inf) < eps:
                        break
                    try:
                        delta_m = -rm[l] * np.linalg.solve(hess_m, grad_m)
                    except np.linalg.LinAlgError as e:
                        print('post_mean', e)
                        continue
                    last_m[:, l] = post_mean[:, l]
                    last_rate[:] = rate
                    predict = np.inner(grad_m, delta_m) + 0.5 * np.dot(delta_m, np.dot(hess_m, delta_m))
                    post_mean[:, l] += delta_m
                    updaterate(range(T), range(N))
                    lb = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, frate=rate)
                    if np.isnan(lb) or lb < lbound[it - 1]:
                        rm[l] *= dec
                        rm[l] += eps
                        post_mean[:, l] = last_m[:, l]
                        rate[:] = last_rate
                    elif lb - lbound[it - 1] > thld * predict:
                        rm[l] *= inc
                        # if rm[l] > 1:
                        #     rm[l] = 1.0
                    post_mean[:, l] -= np.mean(post_mean[:, l])

        # posterior covariance
        if not fixV:
            for l in range(L):
                for t in range(T):
                    last_rate[t, :] = rate[t, :]
                    last_V[:] = post_cov[l, :, :]
                    k_ = post_inv[l, t, t] - 1 / post_cov[l, t, t]  # \tilde{k}_tt
                    old_vtt = post_cov[l, t, t]
                    # fixed point iterations
                    for _ in range(fpinter):
                        post_cov[l, t, t] = 1 / (prior_inv[l, t, t] - k_ + np.sum(rate[t, :] * alpha[l, :] * alpha[l, :]))
                        updaterate([t], range(N))
                    # update post_cov
                    not_t = np.arange(T) != t
                    post_cov[np.ix_([l], not_t, not_t)] = post_cov[np.ix_([l], not_t, not_t)] \
                                                   + (post_cov[l, t, t] - old_vtt) \
                                                   * np.outer(post_cov[l, t, not_t], post_cov[l, t, not_t]) / (old_vtt * old_vtt)
                    post_cov[l, t, not_t] = post_cov[l, not_t, t] = post_cov[l, t, t] * post_cov[l, t, not_t] / old_vtt
                    # update k_tt
                    post_inv[l, t, t] = k_ + 1 / post_cov[l, t, t]
                    # lb = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, rate=rate)
                    # if np.isnan(lb) or lb < lbound[it - 1]:
                    #     # print('post_cov[{}] decreased L.'.format(l))
                    #     post_inv[l, t, t] = k_ + 1 / post_cov[l, t, t]
                    #     post_cov[l, :, :] = last_V
                    #     rate[t, :] = last_rate[t, :]

        for l in range(L):
            prior_cov[l, :, :]

        # update lower bound
        lbound[it] = lowerbound(spike, beta, alpha, prior_mean, prior_inv, post_mean, post_cov, regressor=regressor, frate=rate)

        # check convergence
        del_a = 0.0 if fixa else np.max(np.abs(old_a - alpha))
        del_b = 0.0 if fixb or p == 0 else np.max(np.abs(old_b - beta))
        del_m = 0.0 if fixm else np.max(np.abs(old_m - post_mean))
        del_V = 0.0 if fixV else np.max(np.abs(old_V - post_cov))
        delta = max(del_a, del_b, del_m, del_V)

        if delta < tol:
            convergent = True

        if verbose:
            print('\nIteration[%d]: L = %.5f, inc = %.10f' %
                  (it + 1, lbound[it], lbound[it] - lbound[it - 1]))
            print('change in alpha = %.10f' % del_a)
            print('change in beta = %.10f' % del_b)
            # print('delta gamma = %.10f' % del_c)
            print('change in posterior mean = %.10f' % del_m)
            print('change in posterior covariance = %.10f' % del_V)

        old_a[:] = alpha
        old_b[:] = beta
        old_m[:] = post_mean
        old_V[:] = post_cov

        it += 1

    if it == maxiter:
        warnings.warn('not convergent', RuntimeWarning)

    stop = time.time()

    return post_mean, post_cov, alpha, beta, a0, b0, lbound[:it], stop - start, convergent
