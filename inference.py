import timeit

import numpy as np
from numpy import empty, empty_like, full, zeros, zeros_like, newaxis, tile
from numpy import identity, diag, einsum, inner, trace, exp, sum, mean, var, abs, sqrt
from numpy import inf, finfo, PINF
from scipy import linalg

from constants import *
from util import history


def elbo(data, prior, posterior, param, optim):



def accumulate(accu, grad, decay):
    """adagrad

    Args:
        accu: accumulation matrix
        grad: new gradient
        decay: expoential decay

    Returns:

    """
    # return decay * accu + (1 - decay) * grad ** 2
    return accu + grad ** 2


def inferpost(data, prior, posterior, param, optim):
    """one step update of posterior

    Args:
        data:
        prior:
        posterior:
        param:

    Returns:

    """
    y = data['y']
    h = data['h']
    channel = data['channel']

    chol = prior['chol']

    mu = posterior['mu']
    w = posterior['w']
    v = posterior['v']

    a = param['a']
    b = param['b']
    noise = param['noise']

    accu_grad_mu = optim['accu_grad_mu']
    decay = optim['decay']
    adagrad = optim['adagrad']
    eps = optim['eps']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    T, r, L, M = chol.shape  # time, rank, no latent, no trail
    N = y.shape[1]
    eyer = identity(r)
    R = empty((T, N), dtype=float)
    U = empty((T, N), dtype=float)

    for m in range(M):
        for l in range(L):
            eta = mu[:, :, m].dot(a) + einsum('ijk, jk -> ik', h[:, :, :, m], b)
            noise = var(y[:, :, m] - eta, axis=0, ddof=0)
            lam = exp((eta + 0.5 * v[:, :, m].dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
            G = chol[:, :, l, m]
            grad_mu = (y[:, spike, m] - lam[:, spike]).dot(a[l, spike]) + \
                      ((y[:, lfp, m] - eta[:, lfp, m]) /
                       noise[lfp]).dot(a[l, lfp]) - linalg.lstsq(G.T, linalg.lstsq(G, mu[:, l, m])[0])[0]

            accu_grad_mu[:, l, m] = accumulate(accu_grad_mu[:, l, m], grad_mu, decay)

            if adagrad:
                wada = (w[:, l, m] + sqrt(eps + accu_grad_mu[:, l, m])).reshape((T, 1))  # adjusted by adagrad
            else:
                wada = w[:, l, m].reshape((T, 1))
            GTWG = G.T.dot(wada * G)

            R[:, spike, m] = y[:, spike, m] - lam[:, spike]
            R[:, lfp, m] = (y[:, lfp, m] - eta[:, lfp]) / noise[lfp]

            u = G.dot(G.T.dot(R.dot(a[l, :]))) - mu[:, l, m]
            delta_mu = u - G.dot((wada * G).T.dot(u)) + \
                      G.dot(GTWG.dot(linalg.solve(eyer + GTWG, (wada * G).T.dot(u), sym_pos=True)))

            mu[:, l, m] += delta_mu
            mu[:, l, m] -= mean(mu[:, l, m])
            scale = linalg.norm(mu[:, l, m], ord=inf)
            mu[:, l, m] /= scale

        eta = mu[:, :, m].dot(a) + einsum('ijk, jk -> ik', h[:, :, :, m], b)
        noise = var(y[:, :, m] - eta, axis=0, ddof=0)
        lam = exp((eta + 0.5 * v[:, :, m].dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
        U[:, spike, m] = lam[:, spike]
        U[:, lfp, m] = 1 / noise[lfp]
        w[:, :, m] = U.dot(a.T ** 2)
        for l in range(L):
            G = chol[:, :, l, m]
            GTWG = G.T.dot(w[:, l, m].reshape((T, 1)) * G)
            v[:, l, m] = sum(G * (G - G.dot(GTWG) + G.dot(GTWG.dot(linalg.solve(eyer + GTWG, GTWG, sym_pos=True)))),
                          axis=1)


def inferparam(data, prior, posterior, param, optim):




def inference(data, prior):
    # initialization
    # factor analysis

    y = data['y']
    L = prior['chol'].shape[2]

    T, N, M = y.shape

    y.reshape((T, N))

    while not converged and i < niter:
        # infer posterior
        inferpost(data, prior, posterior, param, optim)
        # infer parameter
        inferparam(data, prior, posterior, param, optim)

        lb[i] = elbo(data, prior, posterior, param)
        if lb[i] < lb[i - 1]:
            # backtracking
        elif abs(lb[i] - lb[i-1]) < tol * abs(lb[i-1]):
            converged = True
        i += 1

    return posterior