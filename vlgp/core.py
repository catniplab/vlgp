import gc
import logging
import warnings

import numpy as np
from numpy import identity, einsum, trace, empty, diag, var, empty_like, sum, reshape
from numpy.core.umath import sqrt, PINF, log
from numpy.linalg import slogdet
from scipy.linalg import lstsq, eigh, solve, norm, svd, LinAlgError

from .constant import *
from .evaluation import timer
from .gp import gp_small_segments, gp_slice_sampling
from .math import sexp
from .name import *

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
    y_ndim, ntrial, nbin, nreg = model['h'].shape  # neuron, trial, time, regression
    z_ndim = model['mu'].shape[-1]
    prior = model['chol']
    rank = prior[0].shape[-1]

    Ir = identity(rank)

    y = model['y'].reshape((-1, y_ndim))  # concatenate trials
    x = model['h'].reshape((y_ndim, -1, nreg))  # concatenate trials
    y_types = model[Y_TYPE]

    prior = model['chol']

    mu = model['mu'].reshape((-1, z_ndim))
    v = model['v'].reshape((-1, z_ndim))

    a = model['a']
    b = model['b']
    noise = model['noise']

    spike_dims = y_types == SPIKE
    lfp_dims = y_types == LFP

    eta = mu @ a + einsum('ijk, ki -> ji',
                          x.reshape((y_ndim, nbin * ntrial, nreg)), b)
    r = sexp(eta + 0.5 * v @ (a ** 2))
    # possible useless calculation here and for noise when spike and LFP mixed.
    # LFP (Gaussian) has no firing rate and spike (Poisson) has no 'noise'.
    # useless dims could be removed to save computational time and space.

    llspike = sum(y[:, spike_dims] * eta[:, spike_dims] - r[:, spike_dims])
    # verified by predict()

    # noinspection PyTypeChecker
    lllfp = - 0.5 * sum(
        (
            (y[:, lfp_dims] - eta[:, lfp_dims]) ** 2 + v @ (
                a[:, lfp_dims] ** 2)) /
        noise[lfp_dims] + log(noise[lfp_dims]))

    ll = llspike + lllfp

    lb = ll

    eps = 1e-3

    for trial in range(ntrial):
        mu = model['mu'][trial, :]
        w = model['w'][trial, :]
        for z_dim in range(z_ndim):
            G = prior[z_dim]
            GtWG = G.T @ (w[trial, :, z_dim, np.newaxis] * G)
            # TODO: a better approximate of mu^T K^{-1} mu than least squares.
            # G_mldiv_mu = lstsq(G, mu[:, dyn_dim])[0]
            # mu_Kinv_mu = inner(G_mldiv_mu, G_mldiv_mu)

            # mu^T (K + eI)^-1 mu
            mu_Kinv_mu = mu[:, z_dim] @ (
                mu[:, z_dim] - G @ solve(eps * Ir + G.T @ G,
                                         G.T @ mu[:, z_dim],
                                         sym_pos=True)) / eps

            tmp = GtWG @ solve(Ir + GtWG, GtWG,
                               sym_pos=True)  # expected to be nonsingular
            tr = nbin - trace(GtWG) + trace(tmp)
            lndet = slogdet(Ir - GtWG + tmp)[1]

            lb += -0.5 * mu_Kinv_mu - 0.5 * tr + 0.5 * lndet + 0.5 * nbin

    return lb, ll


def leastsq(x, y):
    y_ndim = y.shape[-1]
    p = x.shape[-1]
    x_2d = x.reshape((y_ndim, -1, p))
    y_2d = y.reshape((-1, y_ndim))
    return np.array(
        [lstsq(x_2d[y_dim, :], y_2d[:, y_dim])[0] for y_dim in range(y_ndim)])


def estep(model: dict):
    """Update variational distribution q (E step)"""
    if not model[ESTEP]:
        return

    # See the explanation in mstep.
    constrain_a(model)

    y_ndim = model['y'].shape[-1]
    ntrial, nbin, z_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    a = model['a']
    b = model['b']
    noise = model['noise']
    spike = model[Y_TYPE] == SPIKE
    lfp = model[Y_TYPE] == LFP

    Ir = identity(rank)
    residual = empty((nbin, y_ndim), dtype=float)
    U = empty((nbin, y_ndim), dtype=float)

    y = model['y']
    x = model['h']
    mu = model['mu']
    w = model['w']
    v = model['v']
    dmu = model['dmu']

    for i in range(model['e_niter']):
        # TODO: combine trials
        for trial in range(ntrial):
            xb = einsum('ijk, ki -> ji', x[:, trial, :, :], b)
            eta = mu[trial, :, :] @ a + xb
            r = sexp(eta + 0.5 * v[trial, :, :] @ (a ** 2))
            for z_dim in range(z_ndim):
                G = prior[z_dim]

                # working residuals
                # extensible to many other distributions
                # similar form to GLM
                residual[:, spike] = y[trial, ...][:, spike] - r[:, spike]
                residual[:, lfp] = (y[trial, ...][:, lfp] - eta[:, lfp]) / noise[lfp]

                wadj = w[trial, :, z_dim, np.newaxis]  # keep dimension
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[z_dim, :])) - mu[trial, :, z_dim]
                try:
                    block = solve(Ir + GtWG, (wadj * G).T @ u, sym_pos=True)
                    delta_mu = u - G @ ((wadj * G).T @ u) + G @ (GtWG @ block)
                    clip(delta_mu, model['dmu_bound'])
                except Exception as e:
                    logger.exception(repr(e), exc_info=True)
                    delta_mu = 0

                dmu[trial, :, z_dim] = delta_mu
                mu[trial, :, z_dim] += delta_mu

            eta = mu[trial, :, :] @ a + xb
            r = sexp(eta + 0.5 * v[trial, :, :] @ (a ** 2))
            U[:, spike] = r[:, spike]
            U[:, lfp] = 1 / noise[lfp]
            w[trial, :, :] = U @ (a.T ** 2)
            if model['method'] == 'VB':
                for z_dim in range(z_ndim):
                    G = prior[z_dim]
                    GtWG = G.T @ (w[trial, :, z_dim, np.newaxis] * G)
                    try:
                        block = solve(Ir + GtWG, GtWG, sym_pos=True)
                        v[trial, :, z_dim] = np.sum(
                            G * (G - G @ GtWG + G @ (GtWG @ block)), axis=1)
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

    y_ndim, ntrial, nbin, x_ndim = model[
        'h'].shape  # neuron, trial, time, regression
    ntrial, nbin, z_ndim = model['mu'].shape
    y_types = model[Y_TYPE]

    a = model['a']
    b = model['b']
    da = model['da']
    db = model['db']

    y_2d = model['y'].reshape((-1, y_ndim))  # concatenate trials
    x_2d = model['h'].reshape((y_ndim, -1, x_ndim))  # concatenate trials

    mu_2d = model['mu'].reshape((-1, z_ndim))
    v_2d = model['v'].reshape((-1, z_ndim))

    for i in range(model['m_niter']):
        eta = mu_2d @ a + einsum('ijk, ki -> ji', x_2d, b)
        # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
        r = sexp(eta + 0.5 * v_2d @ (a ** 2))
        model['noise'] = var(y_2d - eta, axis=0, ddof=0)  # MLE

        for y_dim in range(y_ndim):
            if y_types[y_dim] == SPIKE:
                # loading
                mu_plus_v_times_a = mu_2d + v_2d * a[:, y_dim]
                grad_a = mu_2d.T @ y_2d[:, y_dim] - mu_plus_v_times_a.T @ r[:, y_dim]

                if model['hessian']:
                    neghess_a = mu_plus_v_times_a.T @ (r[:, y_dim, np.newaxis] * mu_plus_v_times_a)
                    neghess_a[np.diag_indices_from(neghess_a)] += r[:, y_dim] @ v_2d

                    try:
                        delta_a = solve(neghess_a, grad_a, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_a = model['learning_rate'] * grad_a
                else:
                    delta_a = model['learning_rate'] * grad_a

                clip(delta_a, model['da_bound'])
                da[:, y_dim] = delta_a
                a[:, y_dim] += delta_a

                # regression
                grad_b = x_2d[y_dim, :].T @ (y_2d[:, y_dim] - r[:, y_dim])

                if model['hessian']:
                    neghess_b = x_2d[y_dim, :].T @ (
                        r[:, y_dim, np.newaxis] * x_2d[y_dim, :])
                    try:
                        delta_b = solve(neghess_b, grad_b, sym_pos=True)
                    except Exception as e:
                        logger.exception(repr(e), exc_info=True)
                        delta_b = model['learning_rate'] * grad_b
                else:
                    delta_b = model['learning_rate'] * grad_b

                clip(delta_b, model['db_bound'])
                db[:, y_dim] = delta_b
                b[:, y_dim] += delta_b
            elif y_types[y_dim] == LFP:
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                tmp = mu_2d.T @ mu_2d
                tmp[np.diag_indices_from(tmp)] += sum(v_2d, axis=0)
                a[:, y_dim] = solve(tmp, mu_2d.T @ (
                    y_2d[:, y_dim] - x_2d[y_dim, :] @ b[:, y_dim]),
                                    sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, y_dim] = solve(x_2d[y_dim, :].T @ x_2d[y_dim, :],
                                    x_2d[y_dim, :].T @ (y_2d[:, y_dim] - mu_2d @ a[:, y_dim]),
                                    sym_pos=True)
                b[1:, y_dim] = 0
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
        raise ValueError('Unsupported hyperparameter method')


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
    gc.disable()
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

        converged = norm(dmu) < tol * norm(mu) and norm(da) < tol * norm(a) and norm(db) < tol * norm(b)
        stop = converged

        if stop:
            break

    ##############################
    # end of iterative procedure #
    ##############################
    gc.enable()  # enable gabbage collection


def calc_post_cov(model):
    ntrial, nbin, z_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    w = model['w']
    Ir = identity(rank)
    L = empty((ntrial, z_ndim, nbin, rank))
    for trial in range(ntrial):
        for z_dim in range(z_ndim):
            G = prior[z_dim]
            GtWG = G.T @ (w[trial, :, z_dim, np.newaxis].T * G)
            try:
                tmp = Ir - GtWG + GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)
                # A should be PD but numerically not
            except Exception as e:
                # warnings.warn('Singular matrix. Use least squares instead.')
                logger.exception(repr(e), exc_info=True)
                tmp = Ir - GtWG + GtWG @ lstsq(Ir + GtWG, GtWG)[
                    0]  # least squares
            eigval, eigvec = eigh(tmp)
            eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
            L[trial, z_dim, :] = G @ (eigvec @ diag(sqrt(eigval)))
    model['L'] = L


def clip(delta, lbound, ubound=None):
    if ubound is None:
        assert (lbound > 0)
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(delta, lbound, ubound, out=delta)


def update_w(model):
    obs_ndim, ntrial, nbin, nreg = model['h'].shape
    dyn_ndim = model['mu'].shape[-1]

    spike_dims = model[Y_TYPE] == SPIKE
    lfp_dims = model[Y_TYPE] == LFP

    mu_2d = model['mu'].reshape((-1, dyn_ndim))
    x_2d = model['h'].reshape((obs_ndim, -1, nreg))  # concatenate trials
    v_2d = model['v'].reshape((-1, dyn_ndim))
    shape_w = model['w'].shape

    # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
    eta = mu_2d @ model['a'] + einsum('ijk, ki -> ji', x_2d, model['b'])
    r = sexp(eta + 0.5 * v_2d @ (model['a'] ** 2))
    U = empty_like(r)

    U[:, spike_dims] = r[:, spike_dims]
    U[:, lfp_dims] = 1 / model['noise'][lfp_dims]
    model['w'] = reshape(U @ (model['a'].T ** 2), shape_w)


def update_v(model):
    if model['method'] == VB:
        prior = model['chol']
        rank = prior[0].shape[-1]
        Ir = identity(rank)
        ntrial, nbin, z_ndim = model['mu'].shape

        for trial in range(ntrial):
            w = model['w'][trial, :]
            for z_dim in range(z_ndim):
                G = prior[z_dim]
                GtWG = G.T @ (w[:, z_dim, np.newaxis] * G)
                try:
                    model['v'][trial, :, z_dim] = (
                        G * (G - G @ GtWG + G @ (
                            GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)))).sum(
                        axis=1)
                except LinAlgError:
                    warnings.warn("singular I + G'WG")


def constrain_mu(model):
    if not model['constrain_mu']:
        return

    z_ndim = model['dyn_ndim']
    shape = model['mu'].shape
    mu_2d = model['mu'].reshape((-1, z_ndim))
    mean_over_trials = mu_2d.mean(axis=0, keepdims=True)
    std_over_trials = mu_2d.std(axis=0, keepdims=True)

    if model['constrain_mu'] == 'location' or model['constrain_mu'] == 'both':
        mu_2d -= mean_over_trials
        # compensate bias
        # commented to isolated from changing external variables
        model['b'][0, :] += np.squeeze(mean_over_trials @ model['a'])

    if model['constrain_mu'] == 'scale' or model['constrain_mu'] == 'both':
        mu_2d /= std_over_trials
        # compensate loading
        # commented to isolated from changing external variables
        model['a'] *= std_over_trials.T

    model['mu'] = mu_2d.reshape(shape)


def constrain_a(model):
    if not model['constrain_a']:
        return

    method = model['constrain_a']
    eps = model['eps']

    shape_mu = model['mu'].shape
    mu_2d = model['mu'].reshape((-1, shape_mu[-1]))
    a = model['a']
    if method == 'none':
        return
    if method == 'svd':
        # SVD is not good as above
        # noinspection PyTupleAssignmentBalance
        U, s, Vh = svd(a, full_matrices=False)
        model['mu'] = np.reshape(mu_2d @ a @ Vh.T, shape_mu)
        model['a'] = Vh
    else:
        s = norm(a, ord=method, axis=1, keepdims=True) + eps
        a /= s
        # mu_2d *= s.squeeze()  # compensate latent
        # model['mu'] = mu_2d.reshape(shape_mu)
