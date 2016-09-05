"""
Hyperparameter optimization
"""
import math
import warnings

import numpy as np
from numpy import identity, arange, trace, dstack, diag
from numpy import exp, log, sqrt
from numpy.linalg import slogdet
from numpy.random import choice
from scipy.linalg import lstsq, solve, toeplitz, cholesky, LinAlgError, cho_solve
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial.distance import pdist, squareform


def klprime(theta, sigma, n, mu, M, S, eps=1e-6):
    """Derivative of kl part of ELBO w.r.t. hyperparameter
    Args:
        theta: log of timescale
        sigma: variance
        n: length of time
        mu: posterior mean
        M: precalculated matrix involving mu
        S: correlation matrix
        eps: minimum positive value

    Returns:
        gradient
    """
    omega = exp(theta)
    sigmasq = sigma ** 2
    nseg = M.shape[-1]
    dsq = arange(n) ** 2
    Dsq = toeplitz(dsq)
    k = sigmasq * exp(-omega * dsq)
    K = toeplitz(k) + eps * identity(n)
    KinvKD = solve(K, K * Dsq * omega, sym_pos=True)
    grad = 0.0
    for iseg in range(nseg):
        KinvS = solve(K, S[:, :, iseg], sym_pos=True)
        KinvM = solve(K, M[:, :, iseg], sym_pos=True)
        # grad += trace(solve(KinvM + KinvS - identity(n), KinvKD, sym_pos=True))
        grad += trace(lstsq(KinvM + KinvS - identity(n), KinvKD)[0])
    return grad / 2


def kl(theta, sigma, n, mu, M, S, eps=1e-6):
    """kl part of ELBO
    Args:
        theta: log of timescale
        sigma: variance
        n: length of time
        mu: posterior mean
        M: precalculated matrix involving mu
        S: correlation matrix
        eps: minimum positive value

    Returns:
        function value
    """
    omega = exp(theta)
    sigmasq = sigma ** 2
    nseg = mu.shape[-1]
    dsq = arange(n) ** 2
    k = sigmasq * exp(-omega * dsq)
    K = toeplitz(k) + eps * identity(n)
    div = 0.0
    for iseg in range(nseg):
        KinvS = solve(K, S[:, :, iseg], sym_pos=True)
        muKinvmu = mu[:, iseg] @ solve(K, mu[:, iseg], sym_pos=True)
        div += muKinvmu + trace(KinvS) - slogdet(KinvS)[1] - n
    return div / 2


def learngp(obj, latents=None, **kwargs):
    """Main function learning hyperparameters
    Args:
        obj: inference object
        latents: optional true latent
        **kwargs: optional arguments controlling inference

    Returns:
        optimized hyperparameters
    """
    window = kwargs.get('window', 100)
    nseg = kwargs.get('nseg', 10)
    eps = kwargs.get('eps', 1e-6)

    of = kwargs.get('omega_factor', 5)
    ntrial, ntime, nlatent = obj['mu'].shape
    mu = obj['mu'].reshape((-1, nlatent))
    w = obj['w'].reshape((-1, nlatent))
    sigma = obj['sigma'].copy()
    omega = obj['omega'].copy()
    if not (kwargs['learn_sigma'] or kwargs['learn_omega']):
        return sigma, omega
    if latents is None:
        latents = range(nlatent)
    n = ntrial * ntime - window
    start = choice(arange(n), size=nseg)
    win_mu = dstack([mu[i:i + window, :] for i in start])
    win_w = dstack([w[i:i + window, :] for i in start])
    dsq = arange(window) ** 2
    Dsq = toeplitz(dsq)
    for ilatent in latents:
        C = (1 - eps) * exp(-omega[ilatent] * Dsq) + eps * identity(window)
        # K = sigma[ilatent] ** 2 * exp(-omega[ilatent] * Dsq) + eps * identity(window)
        # S = dstack([K - K @ solve(diag(1 / (eps + win_w[:, ilatent, iseg])) + K, K, sym_pos=True) for iseg in
        #             range(nseg)])
        S = dstack([C - C @ solve(diag(1 / (eps + win_w[:, ilatent, iseg])) + C, C, sym_pos=True) for iseg in
                    range(nseg)])
        if kwargs['learn_sigma']:
            tmp = 0.0
            for iseg in range(nseg):
                tmp += win_mu[:, ilatent, iseg] @ solve(C, win_mu[:, ilatent, iseg], sym_pos=True) + trace(
                    solve(C, S[:, :, iseg], sym_pos=True))
            sigma[ilatent] = sqrt(tmp / (window * nseg))
        # M = dstack([outer(win_mu[:, ilatent, iseg], win_mu[:, ilatent, iseg]) for iseg in range(nseg)])
        # mini = minimize(kl, x0=log(omega[ilatent]), jac=klprime, args=(sigma[ilatent], window, win_mu[:, ilatent, :],
        # M, S, eps))
        if kwargs['learn_omega']:
            mini = minimize_scalar(kl, bounds=(log(omega[ilatent] / of), log(omega[ilatent] * of)),
                                   args=(obj['sigma'][ilatent], window, win_mu[:, ilatent, :], None, S, eps),
                                   method='bounded')
            omega[ilatent] = exp(mini.x)
    return sigma, omega
    # return omega


def kernel(x, params):
    """kernel matrix"""
    sigma2, omega, sigma2n = params
    dists = pdist(x.reshape(-1, 1), metric='sqeuclidean')
    K = exp(- omega * dists)
    K = squareform(K)
    np.fill_diagonal(K, 1)
    dK_dsigma2 = K
    K *= sigma2
    dK_domega = - K * squareform(dists)
    K[np.diag_indices_from(K)] += sigma2n
    dK_dsigma2n = np.eye(K.shape[0])
    dK = np.dstack([dK_dsigma2, dK_domega, dK_dsigma2n])
    return K, dK


def marginal(params, x, y):
    K, dK = kernel(x, params)
    # K[np.diag_indices_from(K)] += sigma2n
    try:
        L = cholesky(K, lower=True)
    except LinAlgError:
        return -np.inf, np.zeros_like(params)

    if y.ndim == 1:
        y = y[:, np.newaxis]

    alpha = cho_solve((L, True), y)

    ll_dims = -0.5 * np.einsum('ik,ik->k', y, alpha)
    ll_dims -= np.log(np.diag(L)).sum()
    ll_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    ll = ll_dims.sum(-1)

    tmp = np.einsum('ik,jk->ijk', alpha, alpha)
    tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
    dll_dims = 0.5 * np.einsum('ijl,ijk->kl', tmp, dK)
    dll = dll_dims.sum(-1)

    return ll, dll


def optim(x, y, params0, bounds):
    def obj_func(params):
        ll, dll = marginal(params, x, y)
        return -ll, -dll

    opt, fval, info = fmin_l_bfgs_b(obj_func, params0, bounds=bounds)
    if info['warnflag'] != 0:
        warnings.warn("fmin_l_bfgs_b terminated abnormally with the state: {}".format(info))
    return opt


def subsample(n, size):
    return np.random.choice(n, size, replace=False)
