"""
Hyperparameter optimization
"""
import warnings

import numpy as np
from numpy import exp, log
from numpy import trace
from scipy.linalg import cholesky, LinAlgError, cho_solve
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import pdist, squareform


def se_kernel(x, params):
    """kernel matrix and derivatives"""
    omega, eps = np.exp(params)  # input parameters are logged for unconstrained optimization

    dists = pdist(x.reshape(-1, 1), metric='sqeuclidean')  # vector of pairwise squared distance
    Dsq = squareform(dists)  # distance matrix
    K = exp(- omega * Dsq)  # kernel matrix
    # dK_dsigma2 = K
    K *= 1.0 - eps  # fix variance = 1 - eps (noise variance)
    dK_dlnomega = - K * Dsq * omega
    K[np.diag_indices_from(K)] += eps
    dK_deps = np.eye(K.shape[0]) * eps
    dK = np.dstack([dK_dlnomega, dK_deps])
    # dK = np.dstack([dK_dsigma2, dK_domega])
    return K, dK


def gpr_marginal(params, mask, *args):
    t, mu, *_ = args
    K, dK = se_kernel(t, params)
    dK *= mask[np.newaxis, np.newaxis, :]
    try:
        L = cholesky(K, lower=True)
    except LinAlgError:
        return -np.inf, np.zeros_like(params)

    if mu.ndim == 1:
        mu = mu[:, np.newaxis]

    alpha = cho_solve((L, True), mu)
    ll_dims = -0.5 * np.einsum('ik,ik->k', mu, alpha)
    ll_dims -= np.log(np.diag(L)).sum()
    ll_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    ll = ll_dims.sum(-1)

    tmp = np.einsum('ik,jk->ijk', alpha, alpha)
    tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
    dll_dims = 0.5 * np.einsum('ijl,ijk->kl', tmp, dK)
    dll = dll_dims.sum(-1)

    return ll, dll


def elbo(params, mask, *args):
    t, mu, Sigma = args
    K, dK = se_kernel(t, params)
    dK *= mask[np.newaxis, np.newaxis, :]
    try:
        L = cholesky(K, lower=True)
    except LinAlgError:
        return -np.inf, np.zeros_like(params)

    Kinv = cho_solve((L, True), np.eye(K.shape[0]))  # K inverse

    if mu.ndim == 1:
        mu = mu[:, np.newaxis]

    alpha = cho_solve((L, True), mu)
    ll_dims = -0.5 * np.einsum('ik,ik->k', mu, alpha)
    tmp = np.einsum('ik,jk->ijk', alpha, alpha)
    tmp -= Kinv[:, :, np.newaxis]

    for i in range(Sigma.shape[-1]):
        KinvSigma = cho_solve((L, True), Sigma[:, :, i])
        ll_dims[i] -= 0.5 * trace(KinvSigma)
        tmp[:, :, i] += KinvSigma @ Kinv

    ll_dims -= np.log(np.diag(L)).sum()
    ll = ll_dims.sum(-1)

    dll_dims = 0.5 * np.einsum('ijl,ijk->kl', tmp, dK)
    dll = dll_dims.sum(-1)

    return ll, dll


def construct_posterior_cov(t, w, hparams):
    while True:
        K, dK = se_kernel(t, hparams)
        try:
            L = cholesky(K, lower=True)
            break
        except LinAlgError:
            # return -np.inf, np.zeros_like(params)
            hparams[1] += log(10)

    Kinv = cho_solve((L, True), np.eye(K.shape[0]))  # K inverse

    if w.ndim == 1:
        w = w[:, np.newaxis]

    Sigma = np.zeros((K.shape[0], K.shape[0], w.shape[-1]))
    for i in range(w.shape[-1]):
        L_Sigmainv = cholesky(Kinv + np.diag(w[:, i]), lower=True)
        Sigma[:, :, i] = cho_solve((L_Sigmainv, True), np.eye(K.shape[0]))

    return Sigma


def optim(obj, t, mu, w, params0, bounds, mask, return_f=False):
    log_param0 = np.log(params0)
    log_bounds = np.log(bounds)

    def obj_func(params):
        if obj == 'ELBO':
            Sigma = construct_posterior_cov(t, w, log_param0)
            ll, dll = elbo(params, mask, t, mu, Sigma)
        elif obj == 'GP':
            ll, dll = gpr_marginal(params, mask, t, mu, None)
        else:
            raise NotImplementedError('not supported objective function')
        return -ll, -dll

    opt, fval, info = fmin_l_bfgs_b(obj_func, log_param0, bounds=log_bounds)
    if info['warnflag'] != 0:
        warnings.warn("fmin_l_bfgs_b terminated abnormally with the state: {}".format(info))
    opt = np.exp(opt)
    if return_f:
        return opt, fval

    return opt


def subsample(n, size, successive=False):
    if successive:
        return np.arange(size) + np.random.randint(n - size)
    return np.random.choice(n, size, replace=False)
