from __future__ import division

__author__ = 'yuan'
import itertools
import warnings

import numpy as np


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

    lbound = np.sum(y * (np.dot(Y, b) + np.dot(m, a)) - rate)

    for l in range(L):
        _, logdet = np.linalg.slogdet(V[l, :, :])  # V is positive definite. So discard the sign.
        lbound += -0.5 * np.dot(m[:, l] - mu[:, l], np.dot(omega[l, :, :], m[:, l] - mu[:, l])) + \
                  -0.5 * np.trace(np.dot(omega[l, :, :], V[l, :, :])) + 0.5 * logdet

    return lbound


def variational(y, mu, sigma, p, omega=None, b0=None, a0=None, maxiter=5, inneriter=5, epsilon=np.finfo(float).eps,
                verbose=False):
    """
    :param y: (T, N), spike trains
    :param mu: (T, L), prior mean
    :param sigma: (L, T, T), prior covariance
    :param omega: (L, T, T), inverse prior covariance
    :param p: order of autoregression
    :param maxiter: maximum number of iterations
    :param epsilon: convergence tolerance
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

    # dimensions
    T, N = y.shape
    _, L = mu.shape

    if omega is None:
        omega = np.empty_like(sigma)
        for l in range(L):
            omega[l, :, :] = np.linalg.inv(sigma[l, :, :])

    # initialize posterior to prior
    V = sigma
    K = omega
    m = mu

    # initialize coefficients by least-squares
    Y = np.zeros((T, 1 + p * N), dtype=float)
    Y[:, 0] = 1
    for t in range(T):
        if t - p >= 0:
            Y[t, 1:] = y[t - p:t, :].flatten()  # vectorized by row
        else:
            Y[t, 1 + (p - t) * N:] = y[:t, :].flatten()
    Y_ = np.hstack((Y, m))
    coeffs = np.linalg.lstsq(Y_, y)[0]  # least-squares calculated for each column of y
    # coeffs = np.zeros((1 + p*N + L, N))

    b = coeffs[:1 + p * N, :]
    a = coeffs[1 + p * N:, :]
    if a0 is not None:
        a = a0
    if b0 is not None:
        b = b0

    # initialize rate matrix
    rate = np.empty((T, N))
    updaterate(range(T), range(N))

    # initialize lower bound
    lbound = np.empty(maxiter, dtype=float)
    lbound.fill(np.NINF)
    lbound[0] = lowerbound(y, Y, rate, mu, omega, m, V, b, a)

    # old values
    old_a = np.copy(a)
    old_b = np.copy(b)
    old_m = np.copy(m)
    old_V = np.copy(V)

    basenorm = np.linalg.norm(old_m) + np.sum(map(lambda l: np.linalg.norm(old_V[l, :, :]), range(L)))

    it = 1
    convergent = False

    while not convergent and it < maxiter:
        # optimize coefficients
        # for _ in range(inneriter):
        #     for n in range(N):
        # optimize b
        # grad_b = np.zeros(1 + p * N)
        # hess_b = np.zeros((1 + p * N, 1 + p * N))
        # for t in range(T):
        #     grad_b = grad_b + (y[t, n] - rate[t, n]) * Y[t, :]
        #     hess_b = hess_b - rate[t, n] * np.outer(Y[t, :], Y[t, :])
        # b[:, n] = b[:, n] - np.linalg.lstsq(hess_b, grad_b)[0]
        # b[:, n] = b[:, n] - np.linalg.solve(hess_b, grad_b)
        # b[:, n] = b[:, n] - np.linalg.solve(hess_b + np.diag(0.3*np.diag(hess_b)), grad_b)
        # updaterate(range(T), range(N))

        # optimize a
        # grad_a = np.zeros(L)
        # hess_a = np.zeros((L, L))
        # for t in range(T):
        #     Vt = np.diag(V[:, t, t])
        #     w = m[t, :] + np.dot(Vt, a[:, n])
        #     # rate[t, n] = saferate(t, n, Y, m, V, b, a)
        #     grad_a = grad_a + y[t, n] * m[t, :] - rate[t, n] * w
        #     hess_a = hess_a - rate[t, n] * (np.outer(w, w) + Vt)
        # # a[:, n] = a[:, n] - np.linalg.lstsq(hess_a, grad_a)[0]
        # # a[:, n] = a[:, n] - np.linalg.solve(hess_a, grad_a)
        # a[:, n] = a[:, n] - np.linalg.solve(hess_a + np.diag(0.3*np.diag(hess_a)), grad_a)
        # updaterate(range(T), range(N))

        # optimize posterior
        for l in range(L):
            # optimize V[l]
            for t in range(T):
                k_ = K[l, t, t] - 1 / V[l, t, t]  # \tilde{k}_tt
                old_vtt = V[l, t, t]
                # fixed point iterations
                for _ in range(inneriter):
                    vtt = 1 / (omega[l, t, t] - k_ + np.dot(rate[t, :], a[l, :] * a[l, :]))
                    V[l, t, t] = np.nan_to_num(vtt)
                    # update rate
                    updaterate([t], range(N))
                # update V
                not_t = np.arange(T) != t
                V[np.ix_([l], not_t, not_t)] = np.nan_to_num(V[np.ix_([l], not_t, not_t)] +
                                                           (V[l, t, t] - old_vtt) *
                                                           np.outer(V[l, t, not_t], V[l, t, not_t]) /
                                                           (old_vtt * old_vtt))
                V[l, t, not_t] = V[l, not_t, t] = np.nan_to_num(V[l, t, t] * V[l, t, not_t] / old_vtt)
                # update k_tt
                K[l, t, t] = np.nan_to_num(k_ + 1 / V[l, t, t])
            # updaterate(range(T), range(N))

            # optimize m[l]
            # for _ in range(inneriter):
            #     grad_m = np.nan_to_num(np.dot(y - rate, a[l, :]) - np.dot(omega[l, :, :], (m[:, l] - mu[:, l])))
            #     hess_m = np.nan_to_num(-np.diag(np.dot(rate, a[l, :] * a[l, :]))) - omega[l, :, :]
            #     # m[:, l] = m[:, l] - np.linalg.lstsq(hess_m, grad_m)[0]
            #     delta = np.nan_to_num(np.linalg.solve(hess_m, grad_m))
            #     m[:, l] = m[:, l] - delta
            #     updaterate(range(T), range(N))

        # update lower bound
        lbound[it] = lowerbound(y, Y, rate, mu, omega, m, V, b, a)

        if verbose:
            print 'iteration %d' % (it + 1), ' lower bound = ', lbound[it]

        # check convergence
        delta = np.linalg.norm(old_a - a) + np.linalg.norm(old_b - b) + np.linalg.norm(old_m - m) \
                + np.sum(map(lambda l: np.linalg.norm(old_V[l, :, :] - V[l, :, :]), range(L)))

        if delta < epsilon * basenorm:
            convergent = True

        old_a[:] = a
        old_b[:] = b
        old_m[:] = m
        old_V[:] = V

        it += 1

    if it == maxiter:
        warnings.warn('not convergent', RuntimeWarning)

    return m, V, b, a, lbound, it


def saferate(t, n, Y, m, V, b, a):
    lograte = np.dot(Y[t, :], b[:, n]) + np.dot(m[t, :], a[:, n]) + 0.5 * np.sum(a[:, n] * a[:, n] * V[:, t, t])
    return np.nan_to_num(np.exp(lograte))
