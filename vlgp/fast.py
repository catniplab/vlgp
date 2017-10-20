"""
Fast version of EM algorithm
===
Cut trials into small segments for EM
Infer latent processes with EM-estimated parameters

Convention
---
sigma: variance, usual sigma squared
omega: timescale, usual 0.5 / tau^2
epsilon: state noise variance
"""
import logging
import math
import itertools

import numpy as np
import scipy as sp


logger = logging.getLogger(__name__)


def cut_trial(trial: dict, length=50):
    """Cut a trial into small segments

    :param trial: a trial
    :param length: maximum length of subtrials
    :return: subtrials
    """
    # length determines the running speed, the smaller the faster, but small substrials lose long-term correlation
    # trial['y'] (timestep, neuron)
    y = trial['y']
    nbin = y.shape[0]
    nsub = math.ceil(nbin / length)  # number of subtrials
    subys = np.array_split(y, nsub, axis=0)

    # the subtrials should inherit the other properties of the trial
    template_trial = trial.copy()
    del template_trial['y']
    subtrials = (dict(y=suby, **template_trial) for suby in subys)

    return subtrials


def cut_trials(trials):
    """Cut trials into small segments

    :param trials: list of trials
    :return: subtrials
    """
    from itertools import chain
    return chain([cut_trial(trial) for trial in trials])


def kernel(x, log_params):
    """

    :param x:
    :param log_params:
    :return:
    """
    from scipy.spatial.distance import pdist, squareform

    sigma, omega, epsilon = np.exp(log_params)

    dist = pdist(x.reshape(-1, 1), metric='sqeuclidean')  # vector of squared pairwise distance
    dist_mat = squareform(dist)  # distance matrix
    kern_mat = np.exp(-omega * dist_mat)  # kernel matrix
    sigma_deriv = kern_mat  # derivative wrt sigma squared
    # K *= 1.0 - eps  # fix variance = 1 - eps (noise variance)
    kern_mat *= sigma
    ln_omega_deriv = -kern_mat * dist_mat * omega
    kern_mat[np.diag_indices_from(kern_mat)] += epsilon
    epsilon_deriv = np.eye(kern_mat.shape[0]) * epsilon
    kern_deriv = (sigma_deriv, ln_omega_deriv, epsilon_deriv)

    return kern_mat, kern_deriv


def learn_gp(model):
    """Learn GP parameters

    :param model:
    :return:
    """

    subtrials = model['subtrials']
    zdim = model['zdim']
    gp_params = model['gp_params']

    sigma, omega, epsilon = gp_params

    # unique trial lengths
    # the subtrials of the same length share the same kernal matrix in order to save space and time
    unique_lengths = np.unique((s.shape[0] for s in subtrials))

    # map optimization to each state dimension
    for l in range(zdim):
        trials = ({'mu': trial['mu'][:, l], 'cov': trial['cov'][:, :, l]} for trial in subtrials)
        params = (sigma[l], omega[l], epsilon[l])
        bounds = ((1e-3, 1),
                  model['omega_bound'],
                  (model['gp_noise'] / 2, model['gp_noise'] * 2))
        mask = [False, True, False]

        sigma_optim, omega_optim, epsilon_optim = optimize(trials, params, bounds, mask, unique_lengths)
        omega[l] = omega_optim
        sigma[l] = sigma_optim
        epsilon[l] = epsilon_optim
        # reconstruct small kernel matrices


def optimize(trials, params, bounds, mask, unique_lengths):
    """Optimize single dimension GP

    :param trials:
    :param params:
    :param bounds:
    :param mask:
    :param unique_lengths:
    :return:
    """
    from functools import partial
    from scipy.optimize import minimize

    obj_func = partial(gp_loss, mask=mask, trials=trials, unique_lengths=unique_lengths)

    # optimization is done on log-scale
    params_optim = params

    try:
        res = minimize(obj_func, np.log(params), bounds=np.log(bounds), jac=True)
        params_optim = np.exp(res.x)
    except Exception as e:
        logger.exception(repr(e), exc_info=True)

    return params_optim


def gp_loss(log_params, mask, trials, unique_lengths):
    """

    :param log_params:
    :param mask:
    :param trials:
    :param unique_lengths:
    :return:
    """
    from scipy.linalg import cholesky, cho_solve
    from scipy.linalg import LinAlgError
    from scipy.sparse import block_diag

    kern_mats = {}
    kern_chols = {}
    kern_invs = {}
    kern_derivs = {}
    for l in unique_lengths:
        kern_mat, kern_deriv = kernel(np.arange(l), log_params)
        kern_deriv[mask] = 0
        try:
            kern_chol = cholesky(kern_mat, lower=True)
        except LinAlgError:
            return -np.inf, np.zeros_like(log_params)

        kern_inv = cho_solve((kern_chol, True), np.eye(kern_mat.shape[0]))  # K inverse

        kern_mats[l] = kern_mat
        kern_chols[l] = kern_chol
        kern_derivs[l] = kern_deriv
        kern_invs[l] = kern_inv

    big_mu = np.concatenate((trial['mu'] for trial in trials))
    big_cov = block_diag((trial['cov'] for trial in trials))

    # big_kern_mat = block_diag((kern_mats[l] for l in unique_lengths))
    big_kern_chol = block_diag((kern_chols[l] for l in unique_lengths))
    big_kern_inv = block_diag((kern_invs[l] for l in unique_lengths))
    big_kern_deriv = block_diag((kern_derivs[l] for l in unique_lengths))

    big_kern_inv_mu = cho_solve((big_kern_chol, True), big_mu)
    big_kern_inv_cov = cho_solve((big_kern_chol, True), big_cov)

    loss = 0.5 * np.inner(big_mu, big_kern_inv_mu) + 0.5 * np.trace(big_kern_inv_cov) + np.trace(np.log(big_kern_chol))

    dloss = -0.5 * (np.outer(big_kern_inv_mu, big_kern_inv_mu) -
                    big_kern_inv +
                    big_kern_inv_cov @ big_kern_inv) @ big_kern_deriv

    return loss, dloss


def mstep(model):
    """

    :param model:
    :return:
    """
    from numpy import einsum
    from scipy.linalg import solve, norm
    from vlgp.math import trunc_exp

    if not model['mstep']:
        return

    # It's more reasonable to constrain the latent before mstep.
    # If the parameters are fixed, there's no need to optimize the posterior.
    # Besides, the constraint modifies the loading and bias.
    # constrain_mu(model)
    config = model['config']
    niter = config['niter']
    tol = config['tol']
    hessian = config['hessian']

    ydim = model['ydim']
    lik = model['likelihood']

    a = model['a']
    b = model['b']
    da = model['da']
    db = model['db']

    subtrials = model['subtrials']

    y = np.concatenate((trial['y'] for trial in subtrials), axis=0)
    x = np.concatenate((trial['x'] for trial in subtrials), axis=0)
    mu = np.concatenate((trial['mu'] for trial in subtrials), axis=0)
    v = np.concatenate((np.diagonal(trial['cov']) for trial in subtrials), axis=0)

    should_stop = False
    i = itertools.count(1)
    while not should_stop:
        # TODO: change regression layout to (neuron, time, variable) * (neuron, variable, 1) -> (neuron, time, 1)
        # matmul implements the semantics of the @ operato
        eta = mu @ a + np.squeeze(x @ b, axis=-1)
        r = trunc_exp(eta + 0.5 * v @ (a ** 2))
        gnoise = np.var(y - eta, axis=0, ddof=0)

        for n in range(ydim):
            if lik[n] == 'poisson':
                # loading
                mu_plus_v_times_a = mu + v * a[:, n]
                grad_a = mu.T @ y[:, n] - mu_plus_v_times_a.T @ r[:, n]

                if hessian:
                    nhess_a = mu_plus_v_times_a.T @ (r[:, n, np.newaxis] * mu_plus_v_times_a)
                    nhess_a[np.diag_indices_from(nhess_a)] += r[:, n] @ v

                    try:
                        delta_a = solve(nhess_a, grad_a, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_a = model['learning_rate'] * grad_a
                else:
                    delta_a = model['learning_rate'] * grad_a

                clip_grad(delta_a, model['da_bound'])
                da[:, n] = delta_a
                a[:, n] += delta_a

                # regression
                grad_b = x[..., n].T @ (y[:, n] - r[:, n])

                if hessian:
                    nhess_b = x[..., n].T @ (r[:, np.newaxis, n] * x[..., n])
                    try:
                        delta_b = solve(nhess_b, grad_b, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_b = model['learning_rate'] * grad_b
                else:
                    delta_b = model['learning_rate'] * grad_b

                clip_grad(delta_b, model['db_bound'])
                db[:, n] = delta_b
                b[:, n] += delta_b
            elif lik[n] == 'gaussian':
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                M = mu.T @ mu
                M[np.diag_indices_from(M)] += sum(v, axis=0)
                a[:, n] = solve(M, mu.T @ (
                    y[:, n] - x[..., n] @ b[:, n]),
                                sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, n] = solve(x[..., n].T @ x[..., n],
                                x[..., n].T @ (y[:, n] - mu @ a[:, n]),
                                sym_pos=True)
                b[1:, n] = 0
                # TODO: only make history filter components zeros
            else:
                pass

        if norm(da) < tol * norm(a) and norm(db) < tol * norm(b):
            should_stop = True

        if next(i) > niter:
            should_stop = True

    model['gnoise'] = gnoise


def clip_grad(grad, bound):
    np.clip(grad, -bound, bound, out=grad)
