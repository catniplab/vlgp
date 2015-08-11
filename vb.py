from __future__ import division

import itertools
import warnings
import time

import numpy as np


def saferate(t, n, Y, m, V, b, a):
    lograte = np.dot(Y[t, :], b[:, n]) + np.dot(m[t, :], a[:, n]) + 0.5 * np.sum(a[:, n] * a[:, n] * V[:, t, t])
    return np.nan_to_num(np.exp(lograte))


def lowerbound(y, Y, rate, mu, omega, m, V, b, a):
    """
    Calculate the lower bound without constant terms
    :param y: (T, N), spike trains
    :param Y: (T, 1 + p*N), vectorized spike history
    :param rate: (T, N), E(E(y|x))
    :param mu: (T, L), prior mean
    :param omega: (L, T, T), prior inverse covariances
    :param m: (T, L), latent posterior mean
    :param V: (L, T, T), latent posterior covariances
    :param b: (1 + p*N, N), coefficients of y
    :param a: (L, N), coefficients of x
    :return lbound: lower bound
    """

    _, L = mu.shape
    T, N = y.shape

    lbound = np.sum(y * (np.dot(Y, b) + np.dot(m, a)) - rate)

    for l in range(L):
        lbound += -0.5 * np.dot(m[:, l] - mu[:, l], np.dot(omega[l, :, :], m[:, l] - mu[:, l])) + \
                  -0.5 * np.trace(np.dot(omega[l, :, :], V[l, :, :])) + 0.5 * np.linalg.slogdet(V[l, :, :])[1]

    return lbound


def variational(y, mu, sigma, p, omega=None,
                a0=None, b0=None, m0=None, V0=None, K0=None,
                r=np.finfo(float).eps, maxiter=5, inneriter=5, tol=np.finfo(float).eps,
                verbose=False):
    """
    :param y: (T, N), spike trains
    :param mu: (T, L), prior mean
    :param sigma: (L, T, T), prior covariance
    :param omega: (L, T, T), inverse prior covariance
    :param p: order of autoregression
    :param maxiter: maximum number of iterations
    :param tol: convergence tolerance
    :return
        m: posterior mean
        V: posterior covariance
        b: coefficients of y
        a: coefficients of x
        lbound: lower bound sequence
        it: number of iterations
    """

    def updaterate(t, n):
        # rate = E(E(y|x))
        for t, n in itertools.product(t, n):
            rate[t, n] = saferate(t, n, Y, m, V, b, a)

    start = time.time()  # time when algorithm starts

    # dimensions
    T, N = y.shape
    _, L = mu.shape

    # identity matrix
    hess_adj_b = r * np.identity(1 + p*N)
    id_a = np.identity(L)
    id_m = np.identity(T)
    r_a = np.finfo(float).eps
    r_m = np.finfo(float).eps

    # calculate inverse of prior covariance if not given
    if omega is None:
        omega = np.empty_like(sigma)
        for l in range(L):
            omega[l, :, :] = np.linalg.inv(sigma[l, :, :])

    # read-only variables, protection from unexpected assignment
    y.setflags(write=0)
    mu.setflags(write=0)
    sigma.setflags(write=0)
    omega.setflags(write=0)

    # initialize args
    # make a copy to avoid changing initial values
    if m0 is None:
        m = mu.copy()
    else:
        m = m0.copy()

    if V0 is None:
        V = sigma.copy()
    else:
        V = V0.copy()

    if K0 is None:
        K = omega.copy()
    else:
        K = np.empty_like(V)
        for l in range(L):
            K[l, :, :] = np.linalg.inv(V[l, :, :])

    if a0 is None:
        a = np.zeros((L, N))
    else:
        a = a0.copy()

    if b0 is None:
        b = np.zeros((1 + p * N, N))
    else:
        b = b0.copy()

    # construct history
    Y = history(y, p)

    # initialize rate matrix, rate = E(E(y|x))
    rate = np.empty_like(y)
    updaterate(range(T), range(N))

    # initialize lower bound
    lbound = np.full(maxiter, np.NINF, dtype=float)
    lbound[0] = lowerbound(y, Y, rate, mu, omega, m, V, b, a)

    # old values
    old_a = np.copy(a)
    old_b = np.copy(b)
    old_m = np.copy(m)
    old_V = np.copy(V)

    it = 1
    convergent = False
    while not convergent and it < maxiter:
        # optimize coefficients
        for n in range(N):
            # beta
            for _ in range(inneriter):
                grad_b = np.zeros(1 + p * N)
                hess_b = np.zeros((1 + p * N, 1 + p * N))
                for t in range(T):
                    grad_b = np.nan_to_num(grad_b + (y[t, n] - rate[t, n]) * Y[t, :])
                    hess_b = np.nan_to_num(hess_b - rate[t, n] * np.outer(Y[t, :], Y[t, :]))
                hess_ = hess_b - hess_adj_b  # diagonal of hessian of beta is possibly zero, add a negative quantity
                b[:, n] = b[:, n] - np.linalg.solve(hess_, grad_b)
                updaterate(range(T), [n])

            # alpha
            for _ in range(inneriter):
                grad_a = np.zeros(L)
                hess_a = np.zeros((L, L))
                for t in range(T):
                    Vt = np.diag(V[:, t, t])
                    w = m[t, :] + np.dot(Vt, a[:, n])
                    grad_a = grad_a + y[t, n] * m[t, :] - rate[t, n] * w
                    hess_a = hess_a - rate[t, n] * (np.outer(w, w) + Vt)
                # lagrange multiplier
                grad_a_lag = grad_a - np.inner(a[:, n], grad_a) * a[:, n]
                hess_a_lag = hess_a - np.inner(a[:, n], grad_a)
                if np.linalg.norm(grad_a_lag, ord=np.inf) < np.finfo(float).eps:
                    break
                backup_a = a[:, n]
                a[:, n] = a[:, n] - np.linalg.solve(hess_a_lag, grad_a_lag)
                updaterate(range(T), [n])
                lb = lowerbound(y, Y, rate, mu, omega, m, V, b, a)
                if np.isnan(lb) or lb < lbound[it - 1]:
                    # Newton-Raphson failed, do line search
                    step = - np.inner(grad_a, grad_a) / np.dot(grad_a, np.dot(hess_a, grad_a))
                    a[:, n] = backup_a + step * grad_a
                    updaterate(range(T), [n])
            # discard new alpha if it decreases the lower bound
            lb = lowerbound(y, Y, rate, mu, omega, m, V, b, a)
            if np.isnan(lb) or lb < lbound[it - 1]:
                if verbose:
                    print 'alpha %d rolled back, lb = %.5f' % (n, lb)
                a[:, n] = old_a[:, n]
                updaterate(range(T), [n])

        # posterior
        for l in range(L):
            # covariance
            for t in range(T):
                k_ = K[l, t, t] - 1 / V[l, t, t]  # \tilde{k}_tt
                old_vtt = V[l, t, t]
                # fixed point iterations
                for _ in range(inneriter):
                    V[l, t, t] = 1 / (omega[l, t, t] - k_ + np.sum(rate[t, :] * a[l, :] * a[l, :]))
                    updaterate([t], range(N))
                # update V
                not_t = np.arange(T) != t
                V[np.ix_([l], not_t, not_t)] = V[np.ix_([l], not_t, not_t)] \
                                               + (V[l, t, t] - old_vtt) \
                                                 * np.outer(V[l, t, not_t], V[l, t, not_t]) / (old_vtt * old_vtt)
                V[l, t, not_t] = V[l, not_t, t] = V[l, t, t] * V[l, t, not_t] / old_vtt
                # update k_tt
                K[l, t, t] = k_ + 1 / V[l, t, t]
            # updaterate(range(T), range(N))
            # roll back
            lb = lowerbound(y, Y, rate, mu, omega, m, V, b, a)
            if np.isnan(lb) or lb < lbound[it - 1]:
                if verbose:
                    print 'V rolled back, lb = %.5f' % lb
                V[l, :, :] = old_V[l, :, :]
                updaterate(range(T), range(N))

            # mean
            for _ in range(inneriter):
                grad_m = np.nan_to_num(np.dot(y - rate, a[l, :]) - np.dot(omega[l, :, :], (m[:, l] - mu[:, l])))
                hess_m = np.nan_to_num(-np.diag(np.dot(rate, a[l, :] * a[l, :]))) - omega[l, :, :]
                if np.linalg.norm(grad_m, ord=np.inf) < np.finfo(float).eps:
                    break
                backup_m = m[:, l]
                m[:, l] = m[:, l] - np.linalg.solve(hess_m, grad_m)
                updaterate(range(T), range(N))
                lb = lowerbound(y, Y, rate, mu, omega, m, V, b, a)
                if np.isnan(lb) or lb < lbound[it - 1]:
                    # Newton-Raphson failed, do line search
                    step = - np.inner(grad_m, grad_m) / np.dot(grad_m, np.dot(hess_m, grad_m))
                    m[:, l] = backup_m + step * grad_m
                    updaterate(range(T), range(N))
            # roll back if lower bound decreased
            lb = lowerbound(y, Y, rate, mu, omega, m, V, b, a)
            if np.isnan(lb) or lb < lbound[it - 1]:
                if verbose:
                    print 'm rolled back, lb = %.5f' % lb
                m[:, l] = old_m[:, l]
                updaterate(range(T), range(N))

        # update lower bound
        lbound[it] = lowerbound(y, Y, rate, mu, omega, m, V, b, a)

        # check convergence
        delta = max(np.max(np.abs(old_a - a)), np.max(np.abs(old_b - b)),
                    np.max(np.abs(old_m - m)), np.max(np.abs(old_V - V)))

        if delta < tol:
            convergent = True

        if verbose:
            print 'Iteration[%d]:' % (it + 1), 'L = %.5f' % lbound[it], 'delta = %.5f' % delta

        old_a[:] = a
        old_b[:] = b
        old_m[:] = m
        old_V[:] = V

        it += 1

    if it == maxiter:
        warnings.warn('not convergent', RuntimeWarning)

    stop = time.time()

    return m, V, b, a, lbound[:it], it, stop - start


def history(y, p):
    T, N = y.shape
    Y = np.zeros((T, 1 + p * N), dtype=float)
    Y[:, 0] = 1
    for t in range(T):
        if t - p >= 0:
            Y[t, 1:] = y[t - p:t, :].flatten()  # vectorized by row
        else:
            Y[t, 1 + (p - t) * N:] = y[:t, :].flatten()
    return Y


