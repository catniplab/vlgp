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

    lbound = np.sum(y * (np.dot(Y, b) + np.dot(m, a))) - rate.sum()

    for l in range(L):
        _, logdet = np.linalg.slogdet(V[l, :, :])  # V is positive definite. So discard the sign.
        lbound += -0.5 * np.dot(m[:, l] - mu[:, l], np.dot(omega[l, :, :], m[:, l] - mu[:, l])) - \
                  0.5 * np.trace(np.dot(omega[l, :, :], V[l, :, :])) + 0.5 * logdet

    return lbound


def variational(y, mu, sigma, p, omega=None, maxiter=5, inneriter=10, epsilon=np.finfo(float).eps, verbose=False):
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

    def updaterate():
        for t, n in itertools.product(range(T), range(N)):
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

    # initialize coeffs by least-squares
    Y = np.zeros((T, 1 + p * N), dtype=float)
    Y[:, 0] = 1
    for t in range(T):
        if t - p >= 0:
            Y[t, 1:] = y[t - p:t, :].flatten()  # vectorized by row
        else:
            Y[t, 1 + (p - t) * N:] = y[:t, :].flatten()
    Y_ = np.hstack((Y, m))
    coeffs = np.linalg.lstsq(Y_, y)[0]  # least-squares solution is calculated for each column of y

    b = coeffs[:1 + p * N, :]
    a = coeffs[1 + p * N:, :]

    # initialize rate
    rate = np.empty((T, N))
    updaterate()

    # initialize lower bound
    lbound = np.empty(maxiter, dtype=float)
    lbound.fill(np.NINF)
    lbound[0] = lowerbound(y, Y, rate, mu, omega, m, V, b, a)

    if verbose:
        print 'iteration 1, lower bound = %d' % lbound[0]

    # old values
    old_a = a
    old_b = b
    old_m = m
    old_V = V

    it = 1
    convergent = False

    while not convergent and it < maxiter:
        # optimize coefficients
        for _ in range(inneriter):
            for n in range(N):
                # optimize b
                grad_b = np.zeros(1 + p * N)
                hess_b = np.zeros((1 + p * N, 1 + p * N))
                for t in range(T):
                    grad_b = np.nan_to_num(grad_b + (y[t, n] - rate[t, n]) * Y[t, :])
                    hess_b = np.nan_to_num(hess_b - rate[t, n] * np.outer(Y[t, :], Y[t, :]))
                # coeffs[:, n] = coeffs[:, n] - np.linalg.solve(hess_b, grad_b)
                # to avoid sigular hessian
                b[:, n] = b[:, n] - np.linalg.lstsq(hess_b, grad_b)[0]

                # optimize a
                grad_a = np.zeros(L)
                hess_a = np.zeros((L, L))
                for t in range(T):
                    Vt = np.diag(V[:, t, t])
                    w = m[t, :] + np.dot(Vt, a[:, n])
                    # rate[t, n] = saferate(t, n, Y, m, V, b, a)
                    grad_a = np.nan_to_num(grad_a + y[t, n] * m[t, :] - rate[t, n] * w)
                    hess_a = np.nan_to_num(hess_a - rate[t, n] * (np.outer(w, w) + Vt))
                a[:, n] = a[:, n] - np.linalg.lstsq(hess_a, grad_a)[0]

                updaterate()

        # optimize posterior
        for l in range(L):
            # optimize V[l]
            for t in range(T):
                k_ = K[l, t, t] - 1/V[l, t, t]  # \tilde{k}_tt
                old_vtt = V[l, t, t]
                # fixed point iterations
                for _ in range(inneriter):
                    V[l, t, t] = np.nan_to_num(1 / (omega[l, t, t] - k_ + np.dot(rate[t, :], a[l, :] * a[l, :])))
                    # update \tilde{k}_tt
                    k_ = np.nan_to_num(K[l, t, t] - 1/V[l, t, t])
                    # udpate Lambda
                    for n in range(N):
                        rate[t, n] = saferate(t, n, Y, m, V, b, a)
                # update V
                mask = np.arange(T) != t
                V[np.ix_([l], mask, mask)] = np.nan_to_num(V[np.ix_([l], mask, mask)] + \
                                             (V[l, t, t] - old_vtt) * \
                                             np.outer(V[l, t, mask], V[l, t, mask]) /\
                                             (old_vtt*old_vtt))
                V[l, t, mask] = V[l, mask, t] = np.nan_to_num(V[l, t, t] * V[l, t, mask] / old_vtt)
                # update k_tt
                K[l, t, t] = np.nan_to_num(k_ + 1/V[l, t, t])

            # optimize m[l]
            for _ in range(inneriter):
                grad_m = np.dot(y - rate, a[l, :]) - np.dot(omega[l, :, :], (m[:, l] - mu[:, l]))
                hess_m = -np.diag(np.dot(rate, a[l, :] * a[l, :])) - omega[l, :, :]
                # Y[:, 1+p*N+l] = Y[:, 1+p*N+l] - np.linalg.solve(hess_m, grad_m)
                m[:, l] = m[:, l] - np.linalg.lstsq(hess_m, grad_m)[0]

            updaterate()

        # update lower bound
        lbound[it] = lowerbound(y, Y, rate, mu, omega, m, V, b, a)

        if verbose:
            print 'iteration %d' % (it + 1), ', lower bound = %d' % lbound[it]

        # check convergence
        # if np.abs(lbound[it] - lbound[it-1]) < epsilon*np.abs(lbound[0]):
        #     convergent = True

        delta = np.linalg.norm(old_a - a) + np.linalg.norm(old_b - b) + np.linalg.norm(old_m - m)
        for l in range(L):
            delta += np.linalg.norm(old_V[l, :, :] - V[l, :, :])

        if delta < epsilon:
            convergent = True

        old_a = a
        old_b = b
        old_m = m
        old_V = V

        it += 1

    if it == maxiter:
        warnings.warn('not convergent', RuntimeWarning)

    return m, V, b, a, lbound, it


def saferate(t, n, Y, m, V, b, a):
    lograte = np.dot(Y[t, :], b[:, n]) + np.dot(m[t, :], a[:, n]) + 0.5 * np.dot(a[:, n] * a[:, n], V[:, t, t])
    return np.nan_to_num(np.exp(lograte))
