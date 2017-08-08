"""
Implementation that allows unequal trial lengths

Trials are iterable, typicall a list of dictionaries or structured array, and each contains
the following mandatory field
    y : observation, typically spike count or lfp
and optional fields such as
    x : regression variable, typically history
    z : latent process
    id : trial id
"""
import logging
import warnings

import numpy as np
from numpy import identity, einsum, var, empty_like, sum, reshape, newaxis
from scipy.linalg import lstsq, solve, norm, svd, LinAlgError

from vlgp.constant import *
from vlgp.gp import gp_small_segments, gp_slice_sampling
from vlgp.math import trunc_exp

logger = logging.getLogger(__name__)


def leastsq(x, y):
    y_dim = y.shape[-1]
    x_dim = x.shape[-2]
    x_2d = x.reshape((-1, x_dim, y_dim))
    y_2d = y.reshape((-1, y_dim))
    b = np.array([lstsq(x_2d[..., n], y_2d[:, n])[0] for n in range(y_dim)])
    return b


def estep(job: dict):
    """Update variational distribution q (E step)"""
    if not job[ESTEP]:
        return

    # See the explanation in mstep.
    constrain_loading(job)

    z_dim = job['z_dim']
    a = job['a']
    b = job['b']
    noise = job['noise']

    poiss = job[LIK] == POISSON
    gauss = job[LIK] == GAUSSIAN

    # boolean indexing creates copies
    # pull indexing out of the loop for performance
    # TODO: rearrange y by likelihood in order to replace bool with slicing
    noise_gauss = noise[gauss]

    for i in range(job['e_niter']):
        for trial in job['trials']:
            y = trial['y']
            x = trial['x']
            mu = trial['mu']
            w = trial['w']
            v = trial['v']
            dmu = trial['dmu']
            residual = trial['residual']
            U = trial['U']

            prior = trial['prior']
            rank = prior[0].shape[-1]
            Ir = identity(rank)

            y_poiss = y[:, poiss]
            y_gauss = y[:, gauss]

            xb = einsum('ijk, jk -> ik', x, b)
            eta = mu @ a + xb
            r = trunc_exp(eta + 0.5 * v @ (a ** 2))

            eta_gauss = eta[:, gauss]
            r_poiss = r[:, poiss]

            for l in range(z_dim):
                G = prior[l]

                # working residuals
                # extensible to many other distributions
                # similar form to GLM

                residual[:, poiss] = y_poiss - r_poiss
                residual[:, gauss] = (y_gauss - eta_gauss) / noise_gauss

                wadj = w[:, l, newaxis]  # keep dimension
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[l, :])) - mu[:, l]
                try:
                    M = solve(Ir + GtWG, (wadj * G).T @ u, sym_pos=True)
                    delta_mu = u - G @ ((wadj * G).T @ u) + G @ (GtWG @ M)
                    clip_grad(delta_mu, job['dmu_bound'])
                except Exception as e:
                    logger.exception(repr(e), exc_info=True)
                    delta_mu = 0

                dmu[:, l] = delta_mu
                mu[:, l] += delta_mu

            eta = mu @ a + xb
            r = trunc_exp(eta + 0.5 * v @ (a ** 2))
            U[:, poiss] = r[:, poiss]
            U[:, gauss] = 1 / noise_gauss
            w = U @ (a.T ** 2)
            if job['method'] == 'VB':
                for l in range(z_dim):
                    G = prior[l]
                    GtWG = G.T @ (w[:, l, newaxis] * G)
                    try:
                        M = solve(Ir + GtWG, GtWG, sym_pos=True)
                        v[:, l] = np.sum(
                            G * (G - G @ GtWG + G @ (GtWG @ M)), axis=1)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)

        # center over all trials if not only infer posterior
        # constrain_mu(job)

        # if norm(dmu) < job['tol'] * norm(mu):
        #     break


def mstep(job: dict):
    """Optimize loading and regression (M step)"""
    if not job[MSTEP]:
        return

    # It's more reasonable to constrain the latent before mstep.
    # If the parameters are fixed, there's no need to optimize the posterior.
    # Besides, the constraint modifies the loading and bias.
    constrain_latent(job)

    lik = job[LIK]
    y_dim = job['y_dim']

    a = job['a']
    b = job['b']
    da = job['da']
    db = job['db']

    # concatenate trials
    # TODO: possibly not to concatenate?
    y_2d = np.concatenate([trial['y'] for trial in job['trials']], axis=0)
    x_2d = np.concatenate([trial['x'] for trial in job['trials']], axis=0)
    mu_2d = np.concatenate([trial['mu'] for trial in job['trials']], axis=0)
    v_2d = np.concatenate([trial['v'] for trial in job['trials']], axis=0)

    for i in range(job['m_niter']):
        eta = mu_2d @ a + einsum('ijk, jk -> ik', x_2d, b)
        # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
        r = trunc_exp(eta + 0.5 * v_2d @ (a ** 2))
        job['noise'] = var(y_2d - eta, axis=0, ddof=0)  # MLE

        for n in range(y_dim):
            if lik[n] == POISSON:
                # loading
                mu_plus_v_times_a = mu_2d + v_2d * a[:, n]
                grad_a = mu_2d.T @ y_2d[:, n] - mu_plus_v_times_a.T @ r[:, n]

                if job['hessian']:
                    nhess_a = mu_plus_v_times_a.T @ (
                        r[:, n, newaxis] * mu_plus_v_times_a)
                    nhess_a[np.diag_indices_from(nhess_a)] += r[:, n] @ v_2d

                    try:
                        delta_a = solve(nhess_a, grad_a, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_a = job['learning_rate'] * grad_a
                else:
                    delta_a = job['learning_rate'] * grad_a

                clip_grad(delta_a, job['da_bound'])
                da[:, n] = delta_a
                a[:, n] += delta_a

                # regression
                grad_b = x_2d[..., n].T @ (y_2d[:, n] - r[:, n])

                if job['hessian']:
                    nhess_b = x_2d[..., n].T @ (r[:, newaxis, n] * x_2d[..., n])
                    try:
                        delta_b = solve(nhess_b, grad_b, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_b = job['learning_rate'] * grad_b
                else:
                    delta_b = job['learning_rate'] * grad_b

                clip_grad(delta_b, job['db_bound'])
                db[:, n] = delta_b
                b[:, n] += delta_b
            elif lik[n] == GAUSSIAN:
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                M = mu_2d.T @ mu_2d
                M[np.diag_indices_from(M)] += sum(v_2d, axis=0)
                a[:, n] = solve(M, mu_2d.T @ (
                    y_2d[:, n] - x_2d[..., n] @ b[:, n]),
                                sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, n] = solve(x_2d[..., n].T @ x_2d[..., n],
                                x_2d[..., n].T @ (y_2d[:, n] - mu_2d @ a[:, n]),
                                sym_pos=True)
                b[1:, n] = 0
                # TODO: only make history filter components zeros
            else:
                pass

        if norm(da) < job['tol'] * norm(a) and norm(db) < job['tol'] * norm(b):
            break


def hstep(job: dict):
    """Optimize hyperparameters"""
    if not job[HSTEP]:
        return

    if job[ITER] % job[HPERIOD] != 0:
        return

    # if model['verbose']:
    #     print('Optimize hyperparameter')

    if job['gp'] == 'cutting':
        gp_small_segments(job)
    elif job['gp'] == 'sampling':
        gp_slice_sampling(job)
    else:
        raise ValueError('Unsupported hyperparameter method')


def clip_grad(grad, lbound, ubound=None):
    if ubound is None:
        assert lbound > 0
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(grad, lbound, ubound, out=grad)


def update_w(job):
    ntrial, nbin, x_dim, y_dim = job['x'].shape
    z_dim = job['mu'].shape[-1]

    poiss = job[LIK] == POISSON
    gauss = job[LIK] == GAUSSIAN

    mu_2d = job['mu'].reshape((-1, z_dim))
    x_2d = job['x'].reshape((-1, x_dim, y_dim))  # concatenate trials
    v_2d = job['v'].reshape((-1, z_dim))
    shape_w = job['w'].shape

    # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
    eta = mu_2d @ job['a'] + einsum('ijk, jk -> ik', x_2d, job['b'])
    r = trunc_exp(eta + 0.5 * v_2d @ (job['a'] ** 2))
    U = empty_like(r)

    U[:, poiss] = r[:, poiss]
    U[:, gauss] = 1 / job['noise'][gauss]
    job['w'] = reshape(U @ (job['a'].T ** 2), shape_w)


def update_v(job):
    if job['method'] == VB:
        prior = job[PRIOR]
        Ir = identity(prior[0].shape[-1])
        ntrial, nbin, z_dim = job['mu'].shape

        for trl in range(ntrial):
            w = job['w'][trl, :]
            for l in range(z_dim):
                G = prior[l]
                GtWG = G.T @ (w[:, l, newaxis] * G)
                try:
                    job['v'][trl, :, l] = (
                        G * (G - G @ GtWG + G @ (
                            GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)))).sum(
                        axis=1)
                except LinAlgError:
                    warnings.warn("singular I + G'WG")


def constrain_latent(job):
    if not job['constrain_mu']:
        return

    mu_shape = job['mu'].shape
    z_dim = mu_shape[-1]
    mu_2d = job['mu'].reshape((-1, z_dim))
    mean_over_trials = mu_2d.mean(axis=0, keepdims=True)
    std_over_trials = mu_2d.std(axis=0, keepdims=True)

    if job['constrain_mu'] == 'location' or job['constrain_mu'] == 'both':
        mu_2d -= mean_over_trials
        # compensate bias
        # commented to isolated from changing external variables
        job['b'][0, :] += np.squeeze(mean_over_trials @ job['a'])

    if job['constrain_mu'] == 'scale' or job['constrain_mu'] == 'both':
        mu_2d /= std_over_trials
        # compensate loading
        # commented to isolated from changing external variables
        job['a'] *= std_over_trials.T

    job['mu'] = mu_2d.reshape(mu_shape)


def constrain_loading(job):
    if not job['constrain_a']:
        return

    method = job['constrain_a']
    eps = job['eps']

    mu_shape = job['mu'].shape
    z_dim = mu_shape[-1]
    mu_2d = job['mu'].reshape((-1, z_dim))
    a = job['a']
    if method == 'none':
        return
    if method == 'svd':
        # SVD is not good as above
        # noinspection PyTupleAssignmentBalance
        U, s, Vh = svd(a, full_matrices=False)
        job['mu'] = np.reshape(mu_2d @ a @ Vh.T, mu_shape)
        job['a'] = Vh
    else:
        s = norm(a, ord=method, axis=1, keepdims=True) + eps
        a /= s
        # mu_2d *= s.squeeze()  # compensate latent
        # model['mu'] = mu_2d.reshape(shape_mu)
