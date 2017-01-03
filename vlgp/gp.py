"""
Hyperparameter optimization
"""
import logging

import numpy as np
from numpy import exp, log
from numpy import trace
from numpy.core.umath import sqrt
from scipy.linalg import cholesky, LinAlgError, cho_solve, lstsq
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import pdist, squareform

from vlgp.name import HOBJ, PRIORICHOL
from .math import ichol_gauss

logger = logging.getLogger(__name__)


def se_kernel(x, params):
    """kernel matrix and derivatives"""
    sigma2, omega, eps = np.exp(params)  # input parameters are logged for unconstrained optimization

    dists = pdist(x.reshape(-1, 1), metric='sqeuclidean')  # vector of pairwise squared distance
    Dsq = squareform(dists)  # distance matrix
    K = exp(- omega * Dsq)  # kernel matrix
    dK_dsigma2 = K
    # K *= 1.0 - eps  # fix variance = 1 - eps (noise variance)
    K *= sigma2
    dK_dlnomega = - K * Dsq * omega
    K[np.diag_indices_from(K)] += eps
    dK_deps = np.eye(K.shape[0]) * eps
    dK = np.dstack([dK_dsigma2, dK_dlnomega, dK_deps])
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

    try:
        opt, fval, info = fmin_l_bfgs_b(obj_func, log_param0, bounds=log_bounds)
    except Exception as e:
        opt = log_param0
        logger.exception(repr(e), exc_info=True)
    opt = np.exp(opt)
    if return_f:
        return opt, fval

    return opt


def subsample(n, size, successive=False):
    if successive:
        return np.arange(size) + np.random.randint(n - size)
    return np.random.choice(n, size, replace=False)


# def slice_sample(f, theta, scale, aux_std, rank):
#     """Slice sampling
#         Murray & Adams 2010 Algorithm 4
#     Parameters
#     ----------
#     f : ndarray
#         latent variable
#     theta : float
#         hyperparameter
#     aux_std : float
#         standard deviation of surrogate variable
#     rank : int
#         rank of incomplete Cholesky
#     """
#     # 1. Draw surrogate data: g ~ N(f, aux_std)
#     g = f + np.random.randn(f.size) * aux_std
#
#     # 2. Compute implied latent variates
#     L = 0.0
#     # eta = cho_solve((L, True), f - m)
#
#     # 3. Randomly cener a bracket
#     v = np.random.random() * scale
#     theta_min = theta - v
#     theta_max = theta_min + scale
#
#     # 4. Draw u ~ Uniform(0, 1)
#     u = np.random.random()
#
#     # 5. Determine threashold
#     # y = u * lik(f) * N(g; 0, prior cov + I* aux_std) * prior(theta)
#
#     # while true:
#     # 6. Draw proposal: theta' ~ Uniform(theta_min, theta_max)
#
#     # 7. Compute function f' = L'eta + m'
#
#     # if lik(f') N(g; 0, prior cov' + S') prior(theta') > y:
#     #    return f', theta'
#     # else if theta' < theta:
#     #    theta_min = theta'
#     # else
#     #    theta_max = theta'
#
# def metro_hast(theta, f, slice_width, n, rank):
#     log_prior_theta = lambda x: 0 if 1e-6 < x < 10 else -np.inf
#
#     params = {}
#     step_out = slice_width > 0
#     slice_width = abs(slice_width)
#     slice_fn = lambda pp, lpstar_min:
#     params = slice_sweep(params, slice_fn, slice_width, step_out)
#     return params['position'], params['ichol']


def slice_sweep(particle, slice_fn, slice_width=1, step_out=True):
    lpstar_min = particle['lpstar'] + np.log(np.random.random())
    x_cur = particle['position']

    rr = np.random.random()
    x_l = x_cur - rr * slice_width
    x_r = x_cur + (1 - rr) * slice_width

    if step_out:
        particle['position'] = x_l
        while True:
            particle = slice_fn(particle, lpstar_min)
            if not particle['on_slice']:
                break
            particle['position'] -= slice_width
        x_l = particle['position']
        particle['position'] = x_r
        while True:
            particle = slice_fn(particle, lpstar_min)
            if not particle['on_slice']:
                break
            particle['position'] += slice_width
        x_r = particle['position']

    while True:
        particle['position'] = np.random.random() * (x_r - x_l) + x_l
        particle = slice_fn(particle, lpstar_min)
        if particle['on_slice']:
            break
        if particle['position'] > x_cur:
            x_r = particle['position']
        elif particle['position'] < x_cur:
            x_l = particle['position']
        else:
            raise ValueError('BUG DETECTED: Shrunk to current position and still not acceptable.')


def gp_slice_sampling(model, slice_width=10):
    options = model['options']
    log_prior_theta = lambda l: 0.0 if options['omega_bound'][0] < l < options['omega_bound'][1] else -np.inf
    ntrial, nbin, z_ndim = model['mu'].shape
    for z_dim in range(z_ndim):
        theta = model['omega'][z_dim]
        params = {'position': theta, 'f': model['mu'][:, :, z_dim], 'sigma': model['sigma'][z_dim]}
        update_params(params, lpstar_min=-np.inf, log_prior_theta=log_prior_theta, n=nbin, rank=model['rank'])
        step_out = slice_width > 0
        slice_sweep(params,
                    slice_fn=lambda pp, lpstar_min: update_params(pp, lpstar_min, log_prior_theta, n=nbin,
                                                                  rank=model['rank']),
                    slice_width=slice_width,
                    step_out=step_out)
        model['omega'][z_dim] = params['position']
        model['chol'][z_dim] = params['ichol']


def update_params(params, lpstar_min, log_prior_theta, n, rank):
    theta = params['position']
    l_prior_theta = log_prior_theta(theta)

    if not np.isfinite(l_prior_theta):
        params['lpstar'] = -np.inf
        params['on_slice'] = False
        return params

    G = ichol_gauss(n, theta, rank)

    # q = cho_solve((G, True), params['f'])
    q = np.concatenate(list(map(lambda x: lstsq(G, x)[0], params['f'])))
    log_prior_factor = -0.5 * q.T @ q - n * np.log(params['sigma'])
    params['lpstar'] = log_prior_factor + l_prior_theta
    params['on_slice'] = params['lpstar'] >= lpstar_min
    params['ichol'] = G
    return params


def gp_small_segments(model):
    options = model['options']

    ntrial, nbin, z_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    mu = model['mu']
    w = model['w']
    seg_len = options['seg_len']
    segment = model['segment']
    # subsample_size = options[TRIALLET]
    # if subsample_size is None:
    #     subsample_size = nbin // 2
    # if subsample_size > nbin:
    #     subsample_size = nbin
    sigma = model['sigma']
    omega = model['omega']
    for z_dim in range(z_ndim):
        # subsample = gp.subsample(nbin, subsample_size)
        hparam0 = (sigma[z_dim] ** 2, omega[z_dim], options['gp_noise'])
        bounds = ((1e-3, 1),
                  options['omega_bound'],
                  (options['gp_noise'] / 2, options['gp_noise'] * 2))
        mask = np.array([0, 1, 0])

        sigmasq, omega_new, _ = optim(options[HOBJ],
                                         np.arange(seg_len),
                                         mu[:, segment, z_dim].reshape(-1, seg_len).T,
                                         w[:, segment, z_dim].reshape(-1, seg_len).T,
                                         hparam0,
                                         bounds,
                                         mask=mask,
                                         return_f=False)
        if not np.any(np.isclose(omega_new, options['omega_bound'])):
            omega[z_dim] = omega_new
        sigma[z_dim] = sqrt(sigmasq)
    model[PRIORICHOL] = np.array(
        [ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim] for dyn_dim in range(z_ndim)])
