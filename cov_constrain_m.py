import itertools
import timeit

import numpy as np
from scipy import linalg

from util import makeregressor, sqexpcov


def firingrate(h, m, v, a, b, lb=-30, ub=30):
    """
    :param h:
    :param m:
    :param v: temporal slice of posterior variances
    :param a:
    :param b:
    :param lb:
    :param ub:
    :return:
    """
    eta_x = h.dot(b) + m.dot(a) + 0.5 * v.dot(a ** 2)
    np.clip(eta_x, lb, ub, out=eta_x)
    return np.exp(eta_x)


# ELBO
def elbo(y, h, m, v, a, b, Sigma):
    T, L = m.shape
    eyeT = np.identity(T)
    eta = h.dot(b) + m.dot(a)
    lam = firingrate(h, m, v, a, b)
    lb = np.sum(y * eta - lam)

    for l in range(L):
        w = lam.dot(a[l, :] ** 2)
        isv = eyeT - linalg.solve(np.diag(1 / w) + Sigma[l, :], Sigma[l, :], sym_pos=True)

        lb += -0.5 * m[:, l].dot(linalg.lstsq(Sigma[l, :], m[:, l])[0]) + \
              -0.5 * np.trace(isv) + \
              -0.5 * np.linalg.slogdet(eyeT + Sigma[l, :].dot(np.diag(w)))[1]

    return lb


def train(y, p, prior_var, prior_scale, a0=None, b0=None, m0=None, hyper=False, niter=100, tol=1e-5, verbose=True):

    def updatev():
        lam = firingrate(h, m, v, alpha, beta)
        for l in range(L):
            w = lam.dot(alpha[l, :] ** 2)
            V = Sigma[l, :].dot(eyeT - linalg.solve(np.diag(1 / w) + Sigma[l, :], Sigma[l, :], sym_pos=True))
            v[:, l] = V.diagonal()
    ###################################################

    # dimensions
    T, N = y.shape
    _, L = m0.shape

    eyeT = np.identity(T)

    # hyperparameters
    prior_var = prior_var.copy()
    prior_scale = prior_scale.copy()

    Sigma = np.empty(shape=(L, T, T), dtype=float)
    for l in range(L):
        Sigma[l, :] = sqexpcov(T, prior_scale[l], prior_var[l])

    # read-only variables, protection from unexpected assignment
    y.setflags(write=0)

    # construct makeregressor
    h = makeregressor(y, p)
    h.setflags(write=0)

    # initialize args
    # make alpha copy to avoid changing initial values
    if m0 is None:
        m0 = np.zeros((T, L), dtype=float)
    m = m0.copy()

    if a0 is None:
        a0 = np.zeros((L, N), dtype=float)
    alpha = a0.copy()

    if b0 is None:
        b0 = linalg.lstsq(h, y)[0]
    beta = b0.copy()

    # initialize rate matrix, rate = E(E(spike|x))
    v = np.ones_like(m0) * prior_var
    # updatev()

    # initialize lower bound
    lbound = np.full(niter, np.finfo(float).min)

    # valid values of parameters from previous iteration
    good_alpha = alpha.copy()
    good_beta = beta.copy()
    good_m = m.copy()
    good_var = prior_var.copy()
    good_scale = prior_scale.copy()

    # adagrad
    decay = 0.9
    eps = 1e-6
    accu_grad_a = np.zeros_like(alpha)

    # Optimization
    lbound[0] = elbo(y, h, m, v, alpha, beta, Sigma)
    it = 1
    converged = False
    start = timeit.default_timer()  # time when algorithm starts
    while not converged and it < niter:
        iter_start = timeit.default_timer()
        if verbose:
            print('\nIteration[{:d}]'.format(it))

        ###########
        # weights #
        ###########
        lam = firingrate(h, m, v, alpha, beta)
        for n in range(N):
            grad_b = h.T.dot(y[:, n] - lam[:, n])
            neg_hess_b = h.T.dot((h.T * lam[:, n]).T)
            try:
                delta_b = linalg.solve(neg_hess_b, grad_b, sym_pos=True)
            except linalg.LinAlgError as e:
                print('b', e)
                continue
            beta[:, n] += delta_b
        updatev()

        #############
        # posterior #
        #############
        good_elbo = elbo(y, h, m, v, alpha, beta, Sigma)
        lam = firingrate(h, m, v, alpha, beta)
        for l in range(L):
            w = lam.dot(alpha[l, :] ** 2)

            u = (y - lam).dot(alpha[l, :])
            # grad_m = u - linalg.lstsq(Sigma[l, :], m[:, l])[0]

            u2 = Sigma[l, :].dot(u) - m[:, l]
            delta_m = linalg.lstsq(eyeT + Sigma[l, :].dot(np.diag(w)), u2)[0]
            m[:, l] = good_m[:, l] + delta_m
            m[:, l] -= np.mean(m[:, l])
            m[:, l] /= linalg.norm(m[:, l], ord=np.inf)
            lb = elbo(y, h, m, v, alpha, beta, Sigma)
            if np.isfinite(lb) and lb > good_elbo:
                # Newton step
                good_elbo = lb
            else:
                m[:, l] = good_m[:, l]

            grad_a = (y - lam).T.dot(m[:, l]) - lam.T.dot(v[:, l]) * alpha[l, :]
            accu_grad_a[l, :] = decay * accu_grad_a[l, :] + (1 - decay) * grad_a ** 2
            neg_diag_Ha = np.sum(lam * ((m[:, l, np.newaxis] + np.outer(v[:, l], alpha[l, :])) ** 2), axis=0) + \
                          lam.T.dot(v[:, l])
            delta_a = grad_a / (np.sqrt(eps + accu_grad_a[l, :]) + neg_diag_Ha)
            alpha[l, :] += delta_a
        updatev()

        # store lower bound
        lbound[it] = elbo(y, h, m, v, alpha, beta, Sigma)

        chg_alpha = np.max(np.abs(good_alpha - alpha))
        chg_beta = np.max(np.abs(good_beta - beta))
        chg_post_mean = np.max(np.abs(good_m - m))
        chg_variance = np.max(np.abs(good_var - prior_var)) if hyper else 0.0
        chg_scale = np.max(np.abs(good_scale - prior_scale)) if hyper else 0.0

        if verbose:
            print('lower bound = {:.5f}\n'
                  'increment = {:.10f}\n'
                  'time = {:.2f}s\n'
                  'change in alpha = {:.10f}\n'
                  'change in beta = {:.10f}\n'
                  'change in posterior mean = {:.10f}\n'
                  'change in prior variance = {:.10f}\n'
                  'change in prior scale = {:.10f}'.format(lbound[it], lbound[it] - lbound[it - 1],
                                                           timeit.default_timer() - iter_start,
                                                           chg_alpha, chg_beta, chg_post_mean,
                                                           chg_variance, chg_scale))

        # converged if the change in ELBO is relatively smaller than a tolerance
        if np.abs(lbound[it] - lbound[it - 1]) < tol * np.abs(lbound[it - 1]) and it % 5 == 0:
            converged = True

        # store current iteration
        good_alpha[:] = alpha
        good_beta[:] = beta
        good_m[:] = m
        good_var[:] = prior_var
        good_scale[:] = prior_scale

        it += 1

    stop = timeit.default_timer()
    return lbound[:it], m, alpha, beta, prior_var, prior_scale, a0, b0, stop - start, converged
