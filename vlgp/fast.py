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
import concurrent.futures
import itertools
import logging
import math

import numpy as np

from vlgp import util
from vlgp.core import constrain_mu

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

    # cut the regressor matrix as well
    x = trial['x']  # supposed to be (neuron, time, variable)
    subxs = np.array_split(x, nsub, axis=1)

    # the subtrials should inherit the other properties of the trial
    template_trial = trial.copy()
    del template_trial['y']
    del template_trial['x']
    fast_trials = (dict(y=suby, x=subx, **template_trial) for suby, subx in zip(subys, subxs))  # generator or list?

    return fast_trials


def cut_trials(trials):
    """Cut trials into small segments

    :param trials: list of trials
    :return: subtrials
    """
    from itertools import chain
    return chain((cut_trial(trial) for trial in trials))


def merge(trials):
    """

    :param trials:
    :return:
    """
    # need this?
    # group by trial numbers if they are kept in fast trials
    raise NotImplementedError()


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


@util.log
def maximization(model):
    """

    :param model:
    :return:
    """
    from scipy import optimize
    from scipy.linalg import solve

    def estimate(y, x, z, v, a, b, lik):
        """

        :param y:
        :param x:
        :param z:
        :param v:
        :param a:
        :param b:
        :param lik:
        :return:
        """
        def poisson_loss(param, y, x, z, v):
            zdim = z.shape[-1]
            a, b = param[:zdim], param[zdim:]
            eta = z @ a + x @ b
            r = np.exp(eta) + 0.5 * v @ a ** 2
            zprime = z + v * a
            loglik = np.sum(eta * y - r)
            grad_a = z.T @ y - zprime.T @ r
            grad_b = x.T @ (y - r)
            grad = np.concatenate([grad_a, grad_b])
            return -loglik, -grad

        if lik == 'poisson':
            zdim = z.shape[-1]
            res = optimize.minimize(poisson_loss, np.concatenate([a, b]), args=(y, x, z, v), jac=True)
            return res.x[:zdim], res.x[zdim:]
        elif lik == 'guassian':
            # a's least squares solution for Gaussian channel
            # (m'm + diag(j'v))^-1 m'(y - Hb)
            mumu = mu.T @ mu
            mumu[np.diag_indices_from(mumu)] += sum(v, axis=0)
            a = solve(mumu, mu.T @ (y - x @ b), sym_pos=True)

            # b's least squares solution for Gaussian channel
            # (H'H)^-1 H'(y - ma)
            b = solve(x.T @ x, x.T @ (y - mu @ a), sym_pos=True)
            # history filer doesn't make sense for Gaussian case
            return a, b
        else:
            raise NotImplementedError(lik)

    if not model['mstep']:
        return

    #######################
    # Constrain Posterior #
    #######################
    # Constraining posterior affects the loading matrix and the neuronal bias (while centering the posterior mean).
    # log(Ey) = Az + b
    # Either the loading matrix or posterior mean need to be scaled.
    # It's more convenient to constrain only the posterior so that the loading matrix can be freely estimated.
    # If z is not centered, z' = z - m where m = \bar{z}
    # log(Ey) = A(z' + m) A + b = Az' + Am + b = Az' + b' where b' = Az' + b
    # It's OK not to center z since Az' = b - b' has unique solution at most.
    constrain_mu(model)
    #######################

    lik = model['likelihood']

    a = model['a']
    b = model['b']

    subtrials = model['subtrials']

    y = np.concatenate((trial['y'] for trial in subtrials), axis=0)
    x = np.concatenate((trial['x'] for trial in subtrials), axis=0)
    mu = np.concatenate((trial['mu'] for trial in subtrials), axis=0)
    v = np.concatenate((np.diagonal(trial['cov']) for trial in subtrials), axis=0)

    map(estimate, zip(y.T, x, a.T, b, lik))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(estimate, zip(y.T, x, a.T, b, lik))


@util.log
def expectation(model):
    """

    :param model:
    :return:
    """
    # trial by trial

    def posterior(trial):
        # compute posterior in-place
        pass

    map(posterior, model['subtrials'])  # parallelable

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(posterior, model['subtrials'])

    pass


def clip_grad(grad, bound):
    np.clip(grad, -bound, bound, out=grad)
