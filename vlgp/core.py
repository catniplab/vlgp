"""
Symbols
-------
y : observation, spike count or LFP
x : regression variable, e.g. spike history
z : latent
mu : mean of z
"""
import gc
import logging
import warnings

import numpy as np
from numpy import identity, einsum, trace, empty, diag, var, empty_like, sum, reshape, sqrt, PINF, log
from numpy.linalg import slogdet
from scipy.linalg import lstsq, eigh, solve, norm, svd, LinAlgError

from .constant import *
from .evaluation import timer
from .gp import gp_small_segments, gp_slice_sampling
from .math import trunc_exp

logger = logging.getLogger(__name__)


def elbo(model):
    """
    Evidence Lower BOund (ELBO)

    Parameters
    ----------
    model : dict

    Returns
    -------
    lb : double
        lower bound
    ll : double
        log likelihood
    """
    # neuron, trial, time, regression
    ntrial, nbin, x_dim, y_dim = model['x'].shape
    z_dim = model['mu'].shape[-1]
    prior = model[PRIOR]
    rank = prior[0].shape[-1]

    Ir = identity(rank)

    y_2d = model['y'].reshape((-1, y_dim))  # concatenate trials
    x_2d = model['x'].reshape((-1, x_dim, y_dim))  # concatenate trials
    lik = model[LIK]

    prior = model[PRIOR]

    mu = model['mu'].reshape((-1, z_dim))
    v = model['v'].reshape((-1, z_dim))

    a = model['a']
    b = model['b']
    noise = model['noise']

    poiss = lik == POISSON
    gauss = lik == GAUSSIAN

    # einsum is faster than matmul
    eta = mu @ a + einsum('ijk, jk -> ik', x_2d, b)
    r = trunc_exp(eta + 0.5 * v @ (a ** 2))
    # Possibly useless calculation here.
    # LFP has no firing rate and spike (Poisson) has no extra noise parameter.
    # Unused dims could be removed to save computational time and space.

    llspike = sum(y_2d[:, poiss] * eta[:, poiss] - r[:, poiss])
    # verified by predict()

    # noinspection PyTypeChecker
    lllfp = - 0.5 * sum(
        ((y_2d[:, gauss] - eta[:, gauss]) ** 2 + v @ (a[:, gauss] ** 2)) /
        noise[gauss] + log(noise[gauss]))

    ll = llspike + lllfp

    lb = ll

    eps = 1e-3

    for trl in range(ntrial):
        mu = model['mu'][trl, :]
        w = model['w'][trl, :]
        for l in range(z_dim):
            G = prior[l]
            GtWG = G.T @ (w[trl, :, l, np.newaxis] * G)
            # TODO: a better approximate of mu^T K^{-1} mu than least squares.
            # G_mldiv_mu = lstsq(G, mu[:, dyn_dim])[0]
            # mu_Kinv_mu = inner(G_mldiv_mu, G_mldiv_mu)

            # mu^T (K + eI)^-1 mu
            mu_Kinv_mu = mu[:, l] @ (
                mu[:, l] - G @ solve(eps * Ir + G.T @ G,
                                     G.T @ mu[:, l],
                                     sym_pos=True)) / eps

            # expected to be nonsingular
            M = GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)
            trl = nbin - trace(GtWG) + trace(M)
            lndet = slogdet(Ir - GtWG + M)[1]

            lb += -0.5 * mu_Kinv_mu - 0.5 * trl + 0.5 * lndet + 0.5 * nbin

    return lb, ll


def leastsq(x, y):
    y_dim = y.shape[-1]
    x_dim = x.shape[-2]
    x_2d = x.reshape((-1, x_dim, y_dim))
    y_2d = y.reshape((-1, y_dim))
    b = np.array([lstsq(x_2d[..., n], y_2d[:, n])[0] for n in range(y_dim)])
    return b


def estep(model: dict):
    """Update variational distribution q (E step)"""
    if not model[ESTEP]:
        return

    # See the explanation in mstep.
    constrain_loading(model)

    y_dim = model['y'].shape[-1]
    ntrial, nbin, z_dim = model['mu'].shape
    prior = model[PRIOR]
    rank = prior[0].shape[-1]
    a = model['a']
    b = model['b']
    noise = model['noise']

    Ir = identity(rank)
    residual = empty((nbin, y_dim), dtype=float)
    U = empty((nbin, y_dim), dtype=float)

    y = model['y']
    x = model['x']
    mu = model['mu']
    w = model['w']
    v = model['v']
    dmu = model['dmu']

    poiss = model[LIK] == POISSON
    gauss = model[LIK] == GAUSSIAN

    # boolean indexing creates copies
    # pull indexing out of the loop for performance
    # TODO: rearrange y by likelihood in order to replace bool with slicing
    y_poiss = y[:, :, poiss]
    y_gauss = y[:, :, gauss]
    noise_gauss = noise[gauss]

    for i in range(model['e_niter']):
        # TODO: combine trials
        for trl in range(ntrial):
            xb = einsum('ijk, jk -> ik', x[trl, ...], b)
            eta = mu[trl, :, :] @ a + xb
            r = trunc_exp(eta + 0.5 * v[trl, :, :] @ (a ** 2))

            eta_gauss = eta[:, gauss]
            r_poiss = r[:, poiss]

            for l in range(z_dim):
                G = prior[l]

                # working residuals
                # extensible to many other distributions
                # similar form to GLM

                residual[:, poiss] = y_poiss[trl, :, :] - r_poiss
                residual[:, gauss] = (y_gauss[trl, :, :] - eta_gauss) / noise_gauss

                wadj = w[trl, :, l, np.newaxis]  # keep dimension
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[l, :])) - mu[trl, :, l]
                try:
                    M = solve(Ir + GtWG, (wadj * G).T @ u, sym_pos=True)
                    delta_mu = u - G @ ((wadj * G).T @ u) + G @ (GtWG @ M)
                    clip(delta_mu, model['dmu_bound'])
                except Exception as e:
                    logger.exception(repr(e), exc_info=True)
                    delta_mu = 0

                dmu[trl, :, l] = delta_mu
                mu[trl, :, l] += delta_mu

            eta = mu[trl, :, :] @ a + xb
            r = trunc_exp(eta + 0.5 * v[trl, :, :] @ (a ** 2))
            U[:, poiss] = r[:, poiss]
            U[:, gauss] = 1 / noise_gauss
            w[trl, :, :] = U @ (a.T ** 2)
            if model['method'] == 'VB':
                for l in range(z_dim):
                    G = prior[l]
                    GtWG = G.T @ (w[trl, :, l, np.newaxis] * G)
                    try:
                        M = solve(Ir + GtWG, GtWG, sym_pos=True)
                        v[trl, :, l] = np.sum(
                            G * (G - G @ GtWG + G @ (GtWG @ M)), axis=1)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)

        # center over all trials if not only infer posterior
        # constrain_mu(model)

        if norm(dmu) < model['tol'] * norm(mu):
            break


def mstep(model: dict):
    """Optimize loading and regression (M step)"""
    if not model[MSTEP]:
        return

    # It's more reasonable to constrain the latent before mstep.
    # If the parameters are fixed, there's no need to optimize the posterior.
    # Besides, the constraint modifies the loading and bias.
    constrain_mu(model)

    ntrial, nbin, x_dim, y_dim = model['x'].shape
    ntrial, nbin, z_dim = model['mu'].shape
    lik = model[LIK]

    a = model['a']
    b = model['b']
    da = model['da']
    db = model['db']

    y_2d = model['y'].reshape((-1, y_dim))  # concatenate trials
    x_2d = model['x'].reshape((-1, x_dim, y_dim))  # concatenate trials

    mu_2d = model['mu'].reshape((-1, z_dim))
    v_2d = model['v'].reshape((-1, z_dim))

    for i in range(model['m_niter']):
        eta = mu_2d @ a + einsum('ijk, jk -> ik', x_2d, b)
        # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
        r = trunc_exp(eta + 0.5 * v_2d @ (a ** 2))
        model['noise'] = var(y_2d - eta, axis=0, ddof=0)  # MLE

        for n in range(y_dim):
            if lik[n] == POISSON:
                # loading
                mu_plus_v_times_a = mu_2d + v_2d * a[:, n]
                grad_a = mu_2d.T @ y_2d[:, n] - mu_plus_v_times_a.T @ r[:, n]

                if model['hessian']:
                    nhess_a = mu_plus_v_times_a.T @ (
                        r[:, n, np.newaxis] * mu_plus_v_times_a)
                    nhess_a[np.diag_indices_from(nhess_a)] += r[:, n] @ v_2d

                    try:
                        delta_a = solve(nhess_a, grad_a, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_a = model['learning_rate'] * grad_a
                else:
                    delta_a = model['learning_rate'] * grad_a

                clip(delta_a, model['da_bound'])
                da[:, n] = delta_a
                a[:, n] += delta_a

                # regression
                grad_b = x_2d[..., n].T @ (y_2d[:, n] - r[:, n])

                if model['hessian']:
                    nhess_b = x_2d[..., n].T @ (r[:, np.newaxis, n] * x_2d[..., n])
                    try:
                        delta_b = solve(nhess_b, grad_b, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_b = model['learning_rate'] * grad_b
                else:
                    delta_b = model['learning_rate'] * grad_b

                clip(delta_b, model['db_bound'])
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

        # normalize loading by latent and rescale latent
        # constrain_a(model)

        if norm(da) < model['tol'] * norm(a) and norm(db) < model['tol'] * norm(b):
            break


def hstep(model: dict):
    """Optimize hyperparameters"""
    if not model[HSTEP]:
        return

    if model[ITER] % model[HPERIOD] != 0:
        return

    if model['gp'] == 'cutting':
        gp_small_segments(model)
    elif model['gp'] == 'sampling':
        gp_slice_sampling(model)
    else:
        raise ValueError('Unsupported hyperparameter method: {}'.format(model['gp']))


def vem(model, callbacks=None):
    callbacks = callbacks or []
    tol = model['tol']
    niter = model['niter']

    model.setdefault('it', 0)
    model.setdefault('e_elapsed', [])
    model.setdefault('m_elapsed', [])
    model.setdefault('h_elapsed', [])
    model.setdefault('em_elapsed', [])

    model.setdefault('da', np.zeros_like(model['a']))
    model.setdefault('db', np.zeros_like(model['b']))
    model.setdefault('dmu', np.zeros_like(model['mu']))

    #######################
    # iterative algorithm #
    #######################

    # disable gabbage collection during the iterative procedure
    for it in range(model['it'], niter):
        model['it'] += 1

        with timer() as em_elapsed:
            ##########
            # E step #
            ##########
            with timer() as estep_elapsed:
                estep(model)

            ##########
            # M step #
            ##########
            with timer() as mstep_elapsed:
                mstep(model)

            ###################
            # hyperparam step #
            ###################
            with timer() as hstep_elapsed:
                hstep(model)

        model['e_elapsed'].append(estep_elapsed())
        model['m_elapsed'].append(mstep_elapsed())
        model['h_elapsed'].append(hstep_elapsed())
        model['em_elapsed'].append(em_elapsed())

        for callback in callbacks:
            try:
                callback(model)
            finally:
                pass

        #####################
        # convergence check #
        #####################
        mu = model['mu']
        a = model['a']
        b = model['b']
        dmu = model['dmu']
        da = model['da']
        db = model['db']

        converged = norm(dmu) < tol * norm(mu) and \
                    norm(da) < tol * norm(a) and \
                    norm(db) < tol * norm(b)

        should_stop = converged

        if should_stop:
            break

    ##############################
    # end of iterative procedure #
    ##############################


def calc_post_cov(model):
    ntrial, nbin, z_dim = model['mu'].shape
    prior = model[PRIOR]
    rank = prior[0].shape[-1]
    w = model['w']
    Ir = identity(rank)
    L = empty((ntrial, z_dim, nbin, rank))
    for trl in range(ntrial):
        for l in range(z_dim):
            G = prior[l]
            GtWG = G.T @ (w[trl, :, l, np.newaxis].T * G)
            try:
                M = Ir - GtWG + GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)
                # A should be PD but numerically not
            except Exception as e:
                # warnings.warn('Singular matrix. Use least squares instead.')
                logger.exception(repr(e), exc_info=True)
                M = Ir - GtWG + GtWG @ lstsq(Ir + GtWG, GtWG)[0]
            eigval, eigvec = eigh(M)
            eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
            L[trl, l, :] = G @ (eigvec @ diag(sqrt(eigval)))
    model['L'] = L


def clip(a, lbound, ubound=None):
    if ubound is None:
        assert lbound > 0
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(a, lbound, ubound, out=a)


def update_w(model):
    ntrial, nbin, x_dim, y_dim = model['x'].shape
    z_dim = model['mu'].shape[-1]

    poiss = model[LIK] == POISSON
    gauss = model[LIK] == GAUSSIAN

    mu_2d = model['mu'].reshape((-1, z_dim))
    x_2d = model['x'].reshape((-1, x_dim, y_dim))  # concatenate trials
    v_2d = model['v'].reshape((-1, z_dim))
    shape_w = model['w'].shape

    # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
    eta = mu_2d @ model['a'] + einsum('ijk, jk -> ik', x_2d, model['b'])
    r = trunc_exp(eta + 0.5 * v_2d @ (model['a'] ** 2))
    U = empty_like(r)

    U[:, poiss] = r[:, poiss]
    U[:, gauss] = 1 / model['noise'][gauss]
    model['w'] = reshape(U @ (model['a'].T ** 2), shape_w)


def update_v(model):
    if model['method'] == VB:
        prior = model[PRIOR]
        Ir = identity(prior[0].shape[-1])
        ntrial, nbin, z_dim = model['mu'].shape

        for trl in range(ntrial):
            w = model['w'][trl, :]
            for l in range(z_dim):
                G = prior[l]
                GtWG = G.T @ (w[:, l, np.newaxis] * G)
                try:
                    model['v'][trl, :, l] = (
                        G * (G - G @ GtWG + G @ (
                            GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)))).sum(
                        axis=1)
                except LinAlgError:
                    warnings.warn("singular I + G'WG")


def constrain_mu(model):
    if not model['constrain_mu']:
        return

    mu_shape = model['mu'].shape
    z_dim = mu_shape[-1]
    mu_2d = model['mu'].reshape((-1, z_dim))
    mean_over_trials = mu_2d.mean(axis=0, keepdims=True)
    std_over_trials = mu_2d.std(axis=0, keepdims=True)

    if model['constrain_mu'] in ('location', 'both'):
        mu_2d -= mean_over_trials
        # compensate bias
        # commented to isolated from changing external variables
        model['b'][0, :] += np.squeeze(mean_over_trials @ model['a'])

    if model['constrain_mu'] in ('scale', 'both'):
        mu_2d /= std_over_trials
        # compensate loading
        # commented to isolated from changing external variables
        model['a'] *= std_over_trials.T

    model['mu'] = mu_2d.reshape(mu_shape)


def constrain_loading(model):
    if not model['constrain_a']:
        return

    method = model['constrain_a']
    eps = model['eps']

    mu_shape = model['mu'].shape
    z_dim = mu_shape[-1]
    mu_2d = model['mu'].reshape((-1, z_dim))
    a = model['a']
    if method == 'none':
        return
    if method == 'svd':
        # SVD is not good as above
        # noinspection PyTupleAssignmentBalance
        U, s, Vh = svd(a, full_matrices=False)
        model['mu'] = np.reshape(mu_2d @ a @ Vh.T, mu_shape)
        model['a'] = Vh
    elif method == 'f':
        s = norm(a, ord='fro')
        a /= s
        model['mu'] *= s
    else:
        s = norm(a, ord=method, axis=1, keepdims=True) + eps
        a /= s
        # mu_2d *= s.squeeze()  # compensate latent
        # model['mu'] = mu_2d.reshape(shape_mu)
