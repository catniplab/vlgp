"""
Optimization code for Gaussian Process
"""
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import pdist, squareform

from .math import ichol_gauss


def elbo(params, mask, *args):
    """ELBO with full posterior covariance matrix"""
    t, mu, post_cov = args
    K, dK = kernel(t, params)
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

    for i in range(post_cov.shape[-1]):
        KinvSigma = cho_solve((L, True), post_cov[:, :, i])
        ll_dims[i] -= 0.5 * np.trace(KinvSigma)
        tmp[:, :, i] += KinvSigma @ Kinv

    ll_dims -= np.log(np.diag(L)).sum()
    ll = ll_dims.sum(-1)

    dll_dims = 0.5 * np.einsum('ijl,ijk->kl', tmp, dK)
    dll = dll_dims.sum(-1)

    return ll, dll


def kernel(x, params):
    """kernel matrix and derivatives"""
    sigmasq, omega, eps = params

    dists = pdist(x.reshape(-1, 1), metric='sqeuclidean')  # vector of pairwise squared distance
    Dsq = squareform(dists)  # distance matrix
    K = np.exp(- omega * Dsq)  # kernel matrix
    dK_dsigmasq = K
    # K *= 1.0 - eps  # fix variance = 1 - eps (noise variance)
    K *= sigmasq
    dK_dlnomega = - K * Dsq * omega
    K[np.diag_indices_from(K)] += eps
    dK_deps = np.eye(K.shape[0]) * eps
    dK = np.dstack([dK_dsigmasq, dK_dlnomega, dK_deps])
    return K, dK


def optimize(trials, params, config):
    """Optimize hyperparameters"""
    zdim = params['zdim']
    rank = params['rank']
    dt = params['dt']  # binwidth, set to 1 temporarily

    # priors
    sigma = params['sigma']
    omega = params['omega']
    gp_noise = params['gp_noise']

    # trials
    mu = np.stack([trial['mu'] for trial in trials])
    w = np.stack([trial['w'] for trial in trials])
    window = config['window']
    t = np.arange(window) * dt  # absolute time

    for l in range(zdim):
        initial = (sigma[l] ** 2, omega[l], gp_noise)
        bounds = ((1e-3, 1), config['omega_bound'], (gp_noise / 2, gp_noise * 2))
        mask = np.array([0, 1, 0])

        # transpose each latent dimension to (window, #trials/segments)
        (sigmasq, omega_new, _), fun = optimze1d(t,
                                                 mu[:, :, l].T,
                                                 w[:, :, l].T,
                                                 initial,
                                                 bounds,
                                                 mask=mask)
        if not np.any(np.isclose(omega_new, config['omega_bound'])):
            omega[l] = omega_new
        sigma[l] = np.sqrt(sigmasq)

    params['sigma'] = sigma
    params['omega'] = omega
    make_cholesky(trials, params, config)


def optimze1d(t, mu, w, params, bounds, mask):
    """Optimize hyperparameters of a single dimension"""
    from scipy.optimize import minimize
    log_params = np.log(params)
    log_bounds = np.log(bounds)

    def obj_func(x):
        expx = np.exp(x)
        post_cov = construct_posterior_cov(t, w, expx)
        ll, dll = elbo(expx, mask, t, mu, post_cov)
        return -ll, -dll

    try:
        res = minimize(obj_func, log_params, jac=True, bounds=log_bounds)
        log_params = res.x
        fun = res.fun
        # opt, fval, info = fmin_l_bfgs_b(obj_func, log_params,
        #                                 bounds=log_bounds)
    finally:
        pass
    params = np.exp(log_params)

    return params, fun


def construct_posterior_cov(t, w, params):
    """Make full posterior covariance matrix for hyperparameter tuning"""
    while True:
        K, dK = kernel(t, params)
        try:
            L = cholesky(K, lower=True)
            break
        except LinAlgError:
            # return -np.inf, np.zeros_like(params)
            params[1] += np.log(10)  # increase omega until Cholesky works

    Kinv = cho_solve((L, True), np.eye(K.shape[0]))  # K inverse

    if w.ndim == 1:
        w = w[:, np.newaxis]

    S = np.zeros((K.shape[0], K.shape[0], w.shape[-1]))  # Sigma
    for i in range(w.shape[-1]):
        L_Sinv = cholesky(Kinv + np.diag(w[:, i]), lower=True)
        S[:, :, i] = cho_solve((L_Sinv, True), np.eye(K.shape[0]))

    return S


def make_cholesky(trials, params, config):
    """Make incomplate Cholesky decomposition"""
    zdim = params['zdim']
    rank = params['rank']
    sigma = params['sigma']
    omega = params['omega']
    lengths = np.array([trial['y'].shape[0] for trial in trials])
    unique_lengths = np.unique(lengths)
    for t in unique_lengths:
        params['cholesky'] = dict()
        params['cholesky'][t] = np.array([ichol_gauss(t, omega[l], rank) * sigma[l] for l in range(zdim)])
