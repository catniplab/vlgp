import itertools
import warnings
import time
import numpy as np
from util import history


def saferate(t, n, Y, m, V, b, a, c):
    lograte = c[n] + np.dot(Y[t, :], b[:, n]) + np.dot(m[t, :], a[:, n]) + 0.5 * np.sum(a[:, n] * a[:, n] * V[:, t, t])
    rate = np.nan_to_num(np.exp(lograte))
    return rate if rate > 0 else np.finfo(np.float).eps


def lowerbound(y, b, a, c, mu, omega, m, V, complete=False, Y=None, rate=None):
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

    lbound = np.sum(y * (c + np.dot(Y, b) + np.dot(m, a)) - rate)

    for l in range(L):
        lbound += -0.5 * np.dot(m[:, l] - mu[:, l], np.dot(omega[l, :, :], m[:, l] - mu[:, l])) + \
                  -0.5 * np.trace(np.dot(omega[l, :, :], V[l, :, :])) + 0.5 * np.linalg.slogdet(V[l, :, :])[1]

    return lbound + 0.5 * np.sum(np.linalg.slogdet(omega)[1]) if complete else lbound


def variational(y, mu, sigma, p, omega=None,
                a0=None, b0=None, c0=None, m0=None, V0=None, K0=None,
                fixa=False, fixb=False, fixc=False, fixm=False, fixV=False, anorm=1.0,
                maxiter=50, inneriter=5, tol=1e-6,
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
            rate[t, n] = saferate(t, n, Y, m, V, b, a, c)

    start = time.time()  # time when algorithm starts

    # epsilon
    eps = 2 * np.finfo(np.float).eps

    # dimensions
    T, N = y.shape
    _, L = mu.shape

    eyeL = np.identity(L)
    eyeN = np.identity(N)
    eyeT = np.identity(T)
    oneT = np.ones(T)
    jayT = np.ones((T, T))
    oneTL = np.ones((T, L))

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
    Y = history(y, p)

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
        a = np.random.randn(L, N)
        a /= np.linalg.norm(a) / anorm
    else:
        a = a0.copy()

    if b0 is None:
        b = np.linalg.lstsq(Y, y)[0]
    else:
        b = b0.copy()
    if p == 0:
        b = np.empty(0)

    if c0 is None:
        c = np.log(np.mean(y, axis=0))
    else:
        c = c0.copy()

    # initialize rate matrix, rate = E(E(y|x))
    rate = np.empty_like(y)
    updaterate(range(T), range(N))

    # initialize lower bound
    lbound = np.full(maxiter, np.NINF)
    lbound[0] = lowerbound(y, b, a, c, mu, omega, m, V, Y=Y, rate=rate)

    # old values
    old_a = a.copy()
    old_b = b.copy()
    old_c = c.copy()
    old_m = m.copy()
    old_V = V.copy()

    # variables for recovery
    b0 = b.copy()
    a0 = a.copy()
    c0 = c.copy()
    m0 = m.copy()
    rate0 = rate.copy()
    Vl0 = np.empty((T, T))

    ra = np.ones(N)
    rb = np.ones(N)
    rm = np.ones(L)
    rc = 1.0
    dec = 0.5
    inc = 1.5
    thld = 0.75

    # gradient and hessian
    grad_a_lag = np.zeros(L + 1)
    hess_a_lag = np.zeros((grad_a_lag.size, grad_a_lag.size))
    lam_a = np.zeros(N)
    lam_a0 = lam_a.copy()

    it = 1
    convergent = False
    while not convergent and it < maxiter:
        if not fixc:
            grad_c = np.sum(y - rate, axis=0)
            hess_c = -np.diag(np.sum(rate, axis=0))
            if np.linalg.norm(grad_c, ord=np.inf) < eps:
                break
            try:
                delta_c = -rc * np.linalg.solve(hess_c, grad_c)
            except np.linalg.LinAlgError:
                continue
            c0[:] = c
            rate0[:] = rate[:]
            predict = np.inner(grad_c, delta_c) + 0.5 * np.dot(delta_c, np.dot(hess_c, delta_c))
            c[:] += delta_c
            updaterate(range(T), range(N))
            lb = lowerbound(y, b, a, c, mu, omega, m, V, Y=Y, rate=rate)
            if np.isnan(lb) or lb - lbound[it - 1] < 0:
                rc *= dec
                rc += eps
                c[:] = c0
                rate[:] = rate0[:]
            elif lb - lbound[it - 1] > thld * predict:
                rc *= inc
                if rc > 1:
                    rc = 1.0

        if p != 0 and not fixb:
            for n in range(N):
                grad_b = np.dot(Y.T, y[:, n] - rate[:, n])
                hess_b = np.dot(Y.T, (Y.T * -rate[:, n]).T)
                if np.linalg.norm(grad_b, ord=np.inf) < eps:
                    break
                try:
                    delta_b = -rb[n] * np.linalg.solve(hess_b, grad_b)
                except np.linalg.LinAlgError:
                    continue
                b0[:, n] = b[:, n]
                rate0[:, n] = rate[:, n]
                predict = np.inner(grad_b, delta_b) + 0.5 * np.dot(delta_b, np.dot(hess_b, delta_b))
                b[:, n] += delta_b
                updaterate(range(T), [n])
                lb = lowerbound(y, b, a, c, mu, omega, m, V, Y=Y, rate=rate)
                if np.isnan(lb) or lb < lbound[it - 1]:
                    rb[n] = dec * rb[n] + eps
                    b[:, n] = b0[:, n]
                    rate[:, n] = rate0[:, n]
                elif lb - lbound[it - 1] > thld * predict:
                    rb[n] *= inc
                    if rb[n] > 1:
                        rb[n] = 1.0
        if not fixa:
            for n in range(N):
                # Vt = V.diagonal(axis1=1, axis2=2)
                # m_ = m + c[:, n]
                # grad_a = np.dot(m_.T, y[:, n] - rate[:, n]) - np.dot(Vt, rate[:, n]) * a[:, n]
                # hess_a = - np.dot(m_.T * rate[:, n], m_) \
                #          - np.dot((Vt.T * a[:, n]).T * rate[:, n], Vt.T * a[:, n]) \
                #          - np.diag(np.dot(Vt, rate[:, n]))
                # print(grad_a, hess_a)
                grad_a = np.zeros(L)
                hess_a = np.zeros((L, L))
                for t in range(T):
                    grad_a += y[t, n] * m[t, :] - rate[t, n] * (m[t, :] + np.dot(np.diag(V[: ,t ,t]), a[:, n]))
                    hess_a += -rate[t, n] * (np.outer(m[t, :] + np.dot(np.diag(V[: ,t ,t]), a[:, n]),
                                                      m[t, :] + np.dot(np.diag(V[: ,t ,t]), a[:, n]))
                                             + np.diag(V[: ,t ,t]))
                # lagrange multiplier
                grad_a_lag[:L] = grad_a + 2 * lam_a[n] * a[:, n]
                grad_a_lag[L:] = np.linalg.norm(a, ord='fro') ** 2 - anorm ** 2
                hess_a_lag[:L, :L] = hess_a + 2 * lam_a[n] * eyeL
                hess_a_lag[:L, L:] = hess_a_lag[L:, :L] = 2 * a[:, n]
                hess_a_lag[L:, L:] = 0
                # If gradient is small enough, stop.
                if np.linalg.norm(grad_a_lag, ord=np.inf) < eps:
                    break
                try:
                    delta_a_lag = -ra[n] * np.linalg.solve(hess_a_lag, grad_a_lag)
                except np.linalg.LinAlgError:
                    continue
                lam_a0[n] = lam_a[n]
                a0[:, n] = a[:, n]
                rate0[:, n] = rate[:, n]
                predict = np.inner(grad_a_lag, delta_a_lag) + 0.5 * np.dot(delta_a_lag, np.dot(hess_a_lag, delta_a_lag))
                a[:, n] += delta_a_lag[:L]
                lam_a[n] += delta_a_lag[L:]
                updaterate(range(T), [n])
                lb = lowerbound(y, b, a, c, mu, omega, m, V, Y=Y, rate=rate)
                if np.isnan(lb) or lb - lbound[it - 1] < 0:
                    ra[n] *= dec
                    ra[n] += eps
                    a[:, n] = a0[:, n]
                    rate[:, n] = rate0[:, n]
                    lam_a[n] = lam_a0[n]
                elif lb - lbound[it - 1] > thld * predict:
                    ra[n] *= inc
                    if ra[n] > 1:
                        ra[n] = 1.0

        # posterior mean
        if not fixm:
            for l in range(L):
                grad_m = np.dot(y - rate, a[l, :]) - np.dot(omega[l, :, :], m[:, l] - mu[:, l])
                hess_m = np.diag(np.dot(-rate, a[l, :] * a[l, :])) - omega[l, :, :]
                if np.linalg.norm(grad_m, ord=np.inf) < eps:
                    break
                try:
                    delta_m = -rm[l] * np.linalg.solve(hess_m, grad_m)
                except np.linalg.LinAlgError:
                    continue
                m0[:, l] = m[:, l]
                rate0[:] = rate
                predict = np.inner(grad_m, delta_m) + 0.5 * np.dot(delta_m, np.dot(hess_m, delta_m))
                m[:, l] += delta_m
                updaterate(range(T), range(N))
                lb = lowerbound(y, b, a, c, mu, omega, m, V, Y=Y, rate=rate)
                if np.isnan(lb) or lb < lbound[it - 1]:
                    rm[l] *= dec
                    rm[l] += eps
                    m[:, l] = m0[:, l]
                    rate[:] = rate0
                elif lb - lbound[it - 1] > thld * predict:
                    rm[l] *= inc
                    if rm[l] > 1:
                        rm[l] = 1.0

        # posterior covariance
        if not fixV:
            for l in range(L):
                for t in range(T):
                    rate0[t, :] = rate[t, :]
                    Vl0[:] = V[l, :, :]
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
                    # lb = lowerbound(y, b, a, mu, omega, m, V, Y=Y, rate=rate)
                    # if np.isnan(lb) or lb < lbound[it - 1]:
                    #     print('V[{}] decreased L.'.format(l))
                        # V[l, :, :] = Vl0
                        # K[l, t, t] = k_ + 1 / V[l, t, t]
                        # rate[t, :] = rate0[t, :]

        # update lower bound
        lbound[it] = lowerbound(y, b, a, c, mu, omega, m, V, Y=Y, rate=rate)

        # check convergence
        del_a = 0.0 if fixa else np.max(np.abs(old_a - a))
        del_b = 0.0 if fixb or p == 0 else np.max(np.abs(old_b - b))
        del_c = 0.0 if fixc else np.max(np.abs(old_c - c))
        del_m = 0.0 if fixm else np.max(np.abs(old_m - m))
        del_V = 0.0 if fixV else np.max(np.abs(old_V - V))
        delta = max(del_a, del_b, del_c, del_m, del_V)

        if delta < tol:
            convergent = True

        if verbose:
            print('\nIteration[%d]: L = %.5f, inc = %.10f' %
                  (it + 1, lbound[it], lbound[it] - lbound[it - 1]))
            print('delta alpha = %.10f' % del_a)
            print('delta beta = %.10f' % del_b)
            print('delta gamma = %.10f' % del_c)
            print('delta m = %.10f' % del_m)
            print('delta V = %.10f' % del_V)

        old_a[:] = a
        old_b[:] = b
        old_c[:] = c
        old_m[:] = m
        old_V[:] = V

        it += 1

    if it == maxiter:
        warnings.warn('not convergent', RuntimeWarning)

    stop = time.time()

    return m, V, a, b, c, lbound[:it], stop - start, convergent


def likelihood(y, x, a, b, c):
    T, N = y.shape
    L, _ = x.shape
    pN, _ = b.shape
    p = pN // N

    Y = history(y, p)

    lograte = np.dot(Y, b) + np.dot(x, a) + c
    return np.sum(y * lograte - np.exp(lograte))