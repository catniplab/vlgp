__author__ = 'yuan'
import itertools
import warnings

import numpy as np


def lowerbound(y, Y, rate, mu, Omega, V, coeffs, p):
    """
    Calculate the lower bound without constant terms
    :param y: T*N, spike trains
    :param Y: T*(1 + p*N + L), vectorized y and posterior mean
    :param rate: T*N, E(E(y|x))
    :param mu: T*L, latent prior mean
    :param Omega: L*T*T, latent prior inverse covariances
    :param m: T*L, latent posterior mean
    :param V: L*T*T, latent posterior covariances
    :param coeffs: coefficients
    :param p: order of autoregression
    :return lbound: lower bound
    """

    T, N = y.shape
    _, L = mu.shape

    lbound = 0.0

    # for t, n in itertools.product(range(T), range(N)):
    #     lbound += y[t, n]*np.dot(Y[t, :], coeffs[:, n])
    lbound = np.sum(y * np.dot(Y, coeffs)) - rate.sum()

    for l in range(L):
        _, logdet = np.linalg.slogdet(V[l, :, :])  # V is positive definite. So discard the sign.
        lbound += 0.5*np.dot(Y[:, 1+p*N+l] - mu[:, l], np.dot(Omega[l, :, :], Y[:, 1+p*N+l] - mu[:, l])) + 0.5*logdet

    return lbound


def variational(y, mu, Sigma, p, Omega=None, maxiter=5, inneriter=5, epsilon=np.finfo(float).eps, verbose=False):
    """
    :param y: T*N, spike trains
    :param mu: T*L, prior mean
    :param Sigma: L*T*T, prior covariance
    :param Omega: L*T*T, inverse prior covariance
    :param p: order of autoregression
    :param maxiter: maximum number of iterations
    :param epsilon: convergence tolerance
    :return
        m: posterior mean
        V: posterior covariance
        coeffs: estimate of coefficients
        lbound: lower bound sequence
        it: number of iterations
    """

    # dimensions
    T, N = y.shape
    _, L = mu.shape

    if Omega is None:
        Omega = np.empty_like(Sigma)
        for l in range(L):
            Omega[l, :, :] = np.linalg.inv(Sigma[l, :, :])

    # initialize posterior to prior
    V = Sigma
    K = Omega

    # initialize coefficients
    # coeffs = np.zeros((1 + p*N + L, N), dtype=float)
    # a = np.empty(L)  # alpha
    # initialize lower bound
    lbound = np.empty(maxiter, dtype=float)
    lbound.fill(np.NINF)
    # lbound[0] = np.NINF  # negative infinity

    # initialize coeffs by least-squares
    Y = np.zeros((T, 1+p*N+L), dtype=float)
    Y[:, 0] = 1
    for t in range(T):
        if t - p >= 0:
            Y[t, 1:1+p*N] = y[t-p:t, :].flatten()  # vectorized by row
        else:
            Y[t, 1+(p-t)*N:1+p*N] = y[:t, :].flatten()
        Y[t, 1+p*N:] = mu[t, :]
    coeffs = np.linalg.lstsq(Y, y)[0]  # least-squares solution is calculated for each column of y

    convergent = False

    old_coeffs = coeffs
    old_m = Y[:, 1+p*N:]
    old_V = V

    # initialize rate
    rate = np.empty((T, N))
    for t, n in itertools.product(range(T), range(N)):
        a = coeffs[1+p*N:, n]
        rate[t, n] = saferate(V, Y, a, coeffs, n, t)

    # calculte with initial values of c, m and V
    lbound[0] = lowerbound(y, Y, rate, mu, Omega, V, coeffs, p)

    # intermediate variables
    V_ = np.zeros((1+p*N+L, 1+p*N+L), dtype=float)

    it = 1
    while not convergent and it < maxiter:
        # optimize coeffs
        for _ in range(inneriter):
            for n in range(N):
                # optimize c[, n]
                a = coeffs[1+p*N:, n]  # extract a_n
                grad_coef = np.zeros(1 + p*N + L)
                hess_coef = np.zeros((1 + p*N + L, 1 + p*N + L))
                for t in range(T):
                    V_[1+p*N:, 1+p*N:] = np.diag(V[:, t, t])
                    rate[t, n] = saferate(V, Y, a, coeffs, n, t)
                    w = Y[t, :] + np.dot(V_, coeffs[:, n])
                    grad_coef = np.nan_to_num(grad_coef + y[t, n]*Y[t, :] - rate[t, n]*w)
                    hess_coef = np.nan_to_num(hess_coef - rate[t, n]*(np.outer(w, w) + V_))
                # coeffs[:, n] = coeffs[:, n] - np.linalg.solve(hess_coef, grad_coef)
                # to avoid sigular hessian
                coeffs[:, n] = coeffs[:, n] - np.linalg.lstsq(hess_coef, grad_coef)[0]

        # optimize posterior
        for l in range(L):
            # optimize V[l]
            for t in range(T):
                k_ = K[l, t, t] - 1/V[l, t, t]  # \tilde{k}_tt
                old_vtt = V[l, t, t]
                # fixed point iterations
                for _ in range(inneriter):
                    V[l, t, t] = np.nan_to_num(1/(Omega[l, t, t] - k_ + np.dot(rate[t, :], coeffs[1+p*N+l, :]*coeffs[1+p*N+l, :])))
                    # update \tilde{k}_tt
                    k_ = np.nan_to_num(K[l, t, t] - 1/V[l, t, t])
                    # udpate Lambda
                    # for n in range(N):
                    #     a = coeffs[1+p*N:, n]
                    #     rate[t, n] = saferate(V, Y, a, coeffs, n, t)
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
                lam_a = np.dot(rate, coeffs[1+p*N+l, :])
                grad_m = np.dot(y, coeffs[1+p*N+l, :]) - lam_a - np.dot(Omega[l, :, :], (Y[:, 1+p*N+l] - mu[:, l]))
                hess_m = -np.outer(lam_a, lam_a) - Omega[l, :, :]
                # Y[:, 1+p*N+l] = Y[:, 1+p*N+l] - np.linalg.solve(hess_m, grad_m)
                Y[:, 1+p*N+l] = Y[:, 1+p*N+l] - np.linalg.lstsq(hess_m, grad_m)[0]
                for t, n in itertools.product(range(T), range(N)):
                    a = coeffs[1+p*N:, n]
                    rate[t, n] = saferate(V, Y, a, coeffs, n, t)

        # update lower bound
        lbound[it] = lowerbound(y, Y, rate, mu, Omega, V, coeffs, p)

        # check convergence
        if np.abs(lbound[it] - lbound[it-1]) < epsilon*np.abs(lbound[0]):
            convergent = True

        # if np.linalg.norm(old_coeffs - coeffs) < epsilon:
        #     convergent = True
        it += 1
        old_coeffs = coeffs
        old_m = Y[:, 1+p*N:]
        old_V = V

    if it == maxiter:
        warnings.warn('not convergent', RuntimeWarning)

    return Y[:, 1+p*N:], V, coeffs, lbound, it


def saferate(V, Y, a, coeffs, n, t):
    lograte = np.dot(Y[t, :], coeffs[:, n]) + 0.5 * np.dot(a * a, V[:, t, t])
    return np.nan_to_num(np.exp(lograte))
