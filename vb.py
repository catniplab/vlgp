import itertools
import warnings
import time
import numpy as np


def saferate(t, n, Y, m, V, b, a):
    lograte = np.dot(Y[t, :], b[:, n]) + np.dot(m[t, :], a[:, n]) + 0.5 * np.sum(a[:, n] * a[:, n] * V[:, t, t])
    rate = np.nan_to_num(np.exp(lograte))
    return rate if rate > 0 else np.finfo(np.float).eps


def lowerbound(y, b, a, mu, omega, m, V, complete=False, Y=None, rate=None):
    """
    Calculate the lower bound
    :param y: (T, N), spike trains
    :param b: (1 + p*N, N), coefficients of y
    :param a: (L, N), coefficients of x
    :param mu: (T, L), prior mean
    :param omega: (L, T, T), prior inverse covariances
    :param m: (T, L), latent posterior mean
    :param V: (L, T, T), latent posterior covariances
    :param complete: compute constant terms
    :param Y: (T, 1 + p*N), vectorized spike history
    :param rate: (T, N), E(E(y|x))
    :return lbound: lower bound
    """

    _, L = mu.shape
    T, N = y.shape

    lbound = np.sum(y * (np.dot(Y, b) + np.dot(m, a)) - rate)

    for l in range(L):
        lbound += -0.5 * np.dot(m[:, l] - mu[:, l], np.dot(omega[l, :, :], m[:, l] - mu[:, l])) + \
                  -0.5 * np.trace(np.dot(omega[l, :, :], V[l, :, :])) + 0.5 * np.linalg.slogdet(V[l, :, :])[1]

    return lbound + 0.5 * np.sum(np.linalg.slogdet(omega)[1]) if complete else lbound


def variational(y, mu, sigma, p, omega=None,
                a0=None, b0=None, m0=None, V0=None, K0=None, intercept=True,
                maxiter=5, inneriter=5, tol=1e-6,
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

    # epsilon
    eps = np.finfo(np.float).eps

    # dimensions
    T, N = y.shape
    _, L = mu.shape

    eyeL = np.identity(L)
    eyeN = np.identity(N)
    eyeT = np.identity(T)
    oneT = np.ones(T)
    jayT = np.ones((T, T))

    # identity matrix
    hess_adj_b = eps * np.identity(intercept + p*N)

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

    # construct history
    Y = history(y, p, intercept)

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

    Y_ = np.concatenate((Y, m), axis=1)
    coef = np.linalg.lstsq(Y_, y)[0]

    if a0 is None:
        a = coef[intercept + p*N:, :]
        if np.linalg.norm(a) > 0:
            a /= np.linalg.norm(a)
        else:
            a = np.random.randn(L, N)
            a /= np.linalg.norm(a)
    else:
        a = a0.copy()

    if b0 is None:
        b = coef[:intercept + p*N, :]
    else:
        b = b0.copy()

    # initialize rate matrix, rate = E(E(y|x))
    rate = np.empty_like(y)
    updaterate(range(T), range(N))

    # initialize lower bound
    lbound = np.full(maxiter, np.NINF)
    lbound[0] = lowerbound(y, b, a, mu, omega, m, V, Y=Y, rate=rate)

    # old values
    old_a = a.copy()
    old_b = b.copy()
    old_m = m.copy()
    old_V = V.copy()

    # variables for recovery
    b0 = b.copy()
    a0 = a.copy()
    m0 = m.copy()
    rate0 = rate.copy()

    ra = np.ones(N)
    rb = np.ones(N)
    rm = np.ones(L)
    dec = 0.5
    inc = 1.5
    thld = 0.75


    it = 1
    convergent = False
    while not convergent and it < maxiter:
        for n in range(N):
            grad_b = np.zeros(intercept + p * N)
            hess_b = np.zeros((intercept + p * N, intercept + p * N))
            for t in range(T):
                grad_b += (y[t, n] - rate[t, n]) * Y[t, :]
                hess_b -= rate[t, n] * np.outer(Y[t, :], Y[t, :])
            # if np.linalg.norm(grad_b, ord=np.inf) < eps:
            #     break
            hess_ = hess_b - hess_adj_b  # Hessain of beta is negative semidefinite. Add a small negative diagonal
            delta_b = -rb[n] * np.linalg.solve(hess_, grad_b)
            # if np.linalg.norm(delta_b, ord=np.inf) < eps:
            #     break
            b0[:, n] = b[:, n]
            rate0[:, n] = rate[:, n]
            predict = np.inner(grad_b, delta_b) + 0.5 * np.dot(delta_b, np.dot(hess_, delta_b))
            print('predicted beta[%d] inc = %.10f' % (n, predict))
            b[:, n] = b[:, n] + delta_b
            updaterate(range(T), [n])
            lb = lowerbound(y, b, a, mu, omega, m, V, Y=Y, rate=rate)
            if np.isnan(lb) or lb < lbound[it - 1]:
                rb[n] = dec * rb[n] + eps
                b[:, n] = b0[:, n]
                rate[:, n] = rate0[:, n]
            elif lb - lbound[it - 1] > thld * predict:
                rb[n] *= inc
                if rb[n] > 1:
                    rb[n] = 1.0

        for n in range(N):
            grad_a = np.zeros(L)
            hess_a = np.zeros((L, L))
            for t in range(T):
                Vt = np.diag(V[:, t, t])
                w = m[t, :] + np.dot(Vt, a[:, n])
                grad_a += y[t, n] * m[t, :] - rate[t, n] * w
                hess_a -= rate[t, n] * (np.outer(w, w) + Vt)
            # lagrange multiplier
            grad_a_lag = grad_a - np.inner(a[:, n], grad_a) * a[:, n]
            hess_a_lag = hess_a - np.inner(a[:, n], grad_a)
            # If gradient is small enough, stop.
            # if np.linalg.norm(grad_a_lag, ord=np.inf) < eps:
            #     break
            delta_a = -ra[n] * np.linalg.solve(hess_a_lag, grad_a_lag)
            # if np.linalg.norm(delta_a, ord=np.inf) < eps:
            #     break
            a0[:, n] = a[:, n]
            rate0[:, n] = rate[:, n]
            predict = np.inner(grad_a_lag, delta_a) + 0.5 * np.dot(delta_a, np.dot(hess_a_lag, delta_a))
            print('predicted alpha[%d] inc = %.10f' % (n, predict))
            a[:, n] = a[:, n] + delta_a
            updaterate(range(T), [n])
            lb = lowerbound(y, b, a, mu, omega, m, V, Y=Y, rate=rate)
            if np.isnan(lb) or lb - lbound[it - 1] < 0:
                ra[n] = dec * ra[n] + eps
                a[:, n] = a0[:, n]
                rate[:, n] = rate0[:, n]
            elif lb - lbound[it - 1] > thld * predict:
                ra[n] *= inc
                if ra[n] > 1:
                    ra[n] = 1.0

        # posterior mean
        for l in range(L):
            grad_m = np.nan_to_num(np.dot(y - rate, a[l, :]) - np.dot(omega[l, :, :], (m[:, l] - mu[:, l])))
            hess_m = np.nan_to_num(-np.diag(np.dot(rate, a[l, :] * a[l, :]))) - omega[l, :, :]
            grad_m = np.dot(eyeT -
                            np.dot(np.outer(oneT, a[l, :]),
                                   np.linalg.solve(T * np.outer(a[l, :], a[l, :]) + T * eps * eyeN,
                                                   np.outer(a[l, :], oneT))), grad_m)
            # if np.linalg.norm(grad_m, ord=np.inf) < eps:
            #     break
            delta_m = -rm[l] * np.linalg.solve(hess_m, grad_m)
            m0[:, l] = m[:, l]
            rate0[:] = rate
            predict = np.inner(grad_m, delta_m) + 0.5 * np.dot(delta_m, np.dot(hess_m, delta_m))
            print('predicted m[%d] inc = %.10f' % (l, predict))
            m[:, l] = m[:, l] + delta_m
            updaterate(range(T), range(N))
            lb = lowerbound(y, b, a, mu, omega, m, V, Y=Y, rate=rate)
            if np.isnan(lb) or lb < lbound[it - 1]:
                rm[l] = dec * rm[l] + eps
                m[:, l] = m0[:, l]
                rate[:] = rate0
            elif lb - lbound[it - 1] > thld * predict:
                rm[l] *= inc
                if rm[l] > 1:
                    rm[l] = 1.0

        # posterior covariance
        for l in range(L):
            rate0[:] = rate
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

        # update lower bound
        lbound[it] = lowerbound(y, b, a, mu, omega, m, V, Y=Y, rate=rate)

        # check convergence
        delta = max(np.max(np.abs(old_a - a)), np.max(np.abs(old_b - b)),
                    np.max(np.abs(old_m - m)), np.max(np.abs(old_V - V)))

        if delta < tol:
            convergent = True

        if verbose:
            print('Iteration[%d]: L = %.5f delta = %.10f' % (it + 1, lbound[it], delta))

        old_a[:] = a
        old_b[:] = b
        old_m[:] = m
        old_V[:] = V

        it += 1

    if it == maxiter:
        warnings.warn('not convergent', RuntimeWarning)

    stop = time.time()

    return m, V, b, a, lbound[:it], stop - start


def history(y, p, intercept):
    T, N = y.shape
    Y = np.zeros((T, 1 + p * N), dtype=float)
    Y[:, 0] = 1
    for t in range(T):
        if t - p >= 0:
            Y[t, 1:] = y[t - p:t, :].flatten()  # vectorized by row
        else:
            Y[t, 1 + (p - t) * N:] = y[:t, :].flatten()
    return Y if intercept else Y[:, 1:]
