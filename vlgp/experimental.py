"""Module that does inference"""
# TODO: remove adjust_hessian option
import gc
import pickle
import time
import warnings
from pprint import pprint

import numpy as np
from numpy import identity, einsum, trace, empty, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, sum, copyto, reshape
from numpy.core.umath import sqrt, PINF, log
from numpy.linalg import slogdet
from scipy.linalg import lstsq, eigh, solve, norm, svd, LinAlgError
from scipy.stats import spearmanr
from sklearn.decomposition import factor_analysis
from tqdm import trange

from vlgp import hyper
from vlgp import name
from vlgp import util
from .constant import *
from .name import *
from .evaluation import timer
from .math import ichol_gauss, subspace, sexp
from .util import add_constant, rotate, lagmat, save


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
    obs_ndim, ntrial, nbin, nreg = model['x'].shape  # neuron, trial, time, regression
    dyn_ndim = model['mu'].shape[-1]
    prior = model['chol']
    rank = prior[0].shape[-1]

    Ir = identity(rank)

    y = model['y'].reshape((-1, obs_ndim))  # concatenate trials
    x = model['x'].reshape((obs_ndim, -1, nreg))  # concatenate trials
    obs_types = model['obs_types']

    prior = model['chol']

    mu = model['mu'].reshape((-1, dyn_ndim))
    v = model['v'].reshape((-1, dyn_ndim))

    a = model['a']
    b = model['b']
    noise = model['noise']

    spike_dims = obs_types == SPIKE
    lfp_dims = obs_types == LFP

    eta = mu @ a + einsum('ijk, ki -> ji', x.reshape((obs_ndim, nbin * ntrial, nreg)), b)
    r = sexp(eta + 0.5 * v @ (a ** 2))
    # possible useless calculation here and for noise when spike and LFP mixed.
    # LFP (Gaussian) has no firing rate and spike (Poisson) has no 'noise'.
    # useless dims could be removed to save computational time and space.

    llspike = sum(y[:, spike_dims] * eta[:, spike_dims] - r[:, spike_dims])  # verified by predict()

    # noinspection PyTypeChecker
    lllfp = - 0.5 * sum(
        ((y[:, lfp_dims] - eta[:, lfp_dims]) ** 2 + v @ (a[:, lfp_dims] ** 2)) / noise[lfp_dims] + log(noise[lfp_dims]))

    ll = llspike + lllfp

    lb = ll

    eps = 1e-3

    for trial in range(ntrial):
        mu = model['mu'][trial, :]
        w = model['w'][trial, :]
        for z_dim in range(dyn_ndim):
            G = prior[z_dim]
            GtWG = G.T @ (w[:, [z_dim]] * G)
            # TODO: Need a better approximate of mu^T K^{-1} mu than least squares.
            # G_mldiv_mu = lstsq(G, mu[:, dyn_dim])[0]
            # mu_Kinv_mu = inner(G_mldiv_mu, G_mldiv_mu)

            # mu^T (K + eI)^-1 mu
            mu_Kinv_mu = mu[:, z_dim] @ (
                mu[:, z_dim] - G @ solve(eps * Ir + G.T @ G, G.T @ mu[:, z_dim], sym_pos=True)) / eps

            tmp = GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)  # expected to be nonsingular
            tr = nbin - trace(GtWG) + trace(tmp)
            lndet = slogdet(Ir - GtWG + tmp)[1]

            lb += -0.5 * mu_Kinv_mu - 0.5 * tr + 0.5 * lndet + 0.5 * nbin

    return lb, ll


def check_model(model):
    from .constant import MODEL_FIELDS, PREREQUISITE_FIELDS, DEFAULT_OPTIONS
    for field in MODEL_FIELDS:
        model.setdefault(field, None)

    missing_fields = [field for field in PREREQUISITE_FIELDS if model.get(field) is None]
    if missing_fields:
        raise ValueError('{} missed'.format(missing_fields))

    for k, v in DEFAULT_OPTIONS.items():
        # If key is in the dictionary, return its value. If not, insert key with a value of default and return default.
        model['options'].setdefault(k, v)

    model['obs_types'] = check_y_type(model['obs_types'])


def initialize(model):
    check_model(model)
    options = model['options']

    y = model['y']
    a = model['a']
    b = model['b']
    mu = model['mu']
    sigma = model['sigma']
    omega = model['omega']

    ntrial, nbin, y_ndim = y.shape
    history_filter = model['history_filter']
    z_ndim = model['dyn_ndim']

    eps = options['eps']

    # make design matrix of regression
    h = empty((y_ndim, ntrial, nbin, 1 + history_filter), dtype=float)
    for obs_dim in range(y_ndim):
        for trial in range(ntrial):
            h[obs_dim, trial, :] = add_constant(lagmat(y[trial, :, obs_dim], lag=history_filter))

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent
    if a is None and mu is None:
        fa = factor_analysis.FactorAnalysis(n_components=z_ndim, svd_method='lapack')
        mu = fa.fit_transform(y.reshape((-1, y_ndim)))
        a = fa.components_

        # constrain loading and center latent
        scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
        a /= scale
        mu *= scale.squeeze()  # compensate latent
        mu -= mu.mean(axis=0)
        mu = mu.reshape((ntrial, nbin, z_ndim))

        # noinspection PyTupleAssignmentBalance
        # U, s, Vh = svd(a, full_matrices=False)
        # mu = np.reshape(mu @ a @ Vh.T, (ntrial, nbin, nlatent))
        # a[:] = Vh
    else:
        if mu is None:
            mu = lstsq(a.T, y.reshape((-1, y_ndim)).T)[0].T.reshape((ntrial, nbin, z_ndim))
        elif a is None:
            a = lstsq(mu.reshape((-1, z_ndim)), y.reshape((-1, y_ndim)))[0]

    # initialize regression
    if b is None:
        b = leastsq(h)

    # initialize noises of LFP
    model['noise'] = var(y.reshape((-1, y_ndim)), axis=0, ddof=0)

    ####################
    # initialize prior #
    ####################

    # make Cholesky of prior
    if model['rank'] is None:
        model['rank'] = nbin // 5
    rank = model['rank']

    prior = np.array([ichol_gauss(nbin, omega[z_dim], rank) * sigma[z_dim] for z_dim in range(z_ndim)])

    # fill model fields
    model['a'] = a
    model['b'] = b
    model['mu'] = mu
    model['w'] = zeros_like(mu, dtype=float)
    model['v'] = zeros_like(mu, dtype=float)
    model['x'] = h
    model['chol'] = prior

    model['dmu'] = zeros_like(model['mu'])
    model['da'] = zeros_like(model['da'])
    model['db'] = zeros_like(model['db'])

    update_w(model)
    update_v(model)


def leastsq(x, y):
    y_ndim = y.shape[-1]
    p = x.shape[-1]
    x_ = x.reshape((y_ndim, -1, p))
    y_ = y.reshape((-1, y_ndim))
    return np.array([lstsq(x_[y_dim, :], y_[:, y_dim])[0] for y_dim in range(y_ndim)])


def estep(model):
    """Update variational distribution q (E step)"""
    options = model['options']

    if not options[ESTEP]:
        return

    y_ndim = model['y'].shape[-1]
    ntrial, nbin, z_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    a = model['a']
    b = model['b']
    noise = model['noise']
    spike_dims = model[Y_TYPE] == SPIKE
    lfp_dims = model[Y_TYPE] == LFP

    Ir = identity(rank)
    residual = empty((nbin, y_ndim), dtype=float)
    U = empty((nbin, y_ndim), dtype=float)

    y = model['y']
    x = model['x']
    mu = model['mu']
    w = model['w']
    v = model['v']
    dmu = model['dmu']

    for i in range(options['e_niter']):
        for trial in range(ntrial):
            xb = einsum('ijk, ki -> ji', x[:, trial, :, :], b)
            eta = mu[trial, :, :] @ a + xb
            r = sexp(eta + 0.5 * v[trial, :, :] @ (a ** 2))
            for z_dim in range(z_ndim):
                G = prior[z_dim]

                # working residuals
                # extensible to many other distributions
                # similar form to GLM
                residual[:, spike_dims] = y[trial, :, spike_dims] - r[:, spike_dims]
                residual[:, lfp_dims] = (y[trial, :, lfp_dims] - eta[:, lfp_dims]) / noise[lfp_dims]

                wadj = w[trial, :, [z_dim]]  # keep dimension
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[z_dim, :])) - mu[trial, :, z_dim]
                delta_mu = u - G @ ((wadj * G).T @ u) + \
                           G @ (GtWG @ solve(Ir + GtWG, (wadj * G).T @ u, sym_pos=True))

                clip(delta_mu, options['dmu_bound'])
                dmu[trial, :, z_dim] = delta_mu
                mu[trial, :, z_dim] += delta_mu

            eta = mu[trial, :, :] @ a + xb
            r = sexp(eta + 0.5 * v[trial, :, :] @ (a ** 2))
            U[:, spike_dims] = r[:, spike_dims]
            U[:, lfp_dims] = 1 / noise[lfp_dims]
            w[trial, :, :] = U @ (a.T ** 2)
            if options['method'] == 'VB':
                for z_dim in range(z_ndim):
                    G = prior[z_dim]
                    GtWG = G.T @ (w[trial, :, [z_dim]] * G)
                    try:
                        v[trial, :, z_dim] = (
                        G * (G - G @ GtWG + G @ (GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)))).sum(
                            axis=1)
                    except LinAlgError:
                        warnings.warn("singular I + G'WG")

        # center over all trials if not only infer posterior
        constrain_mu(model)

        if norm(dmu) < options['tol'] * norm(mu):
            break


def mstep(model):
    """Optimize loading and regression (M step)"""
    options = model['options']

    if not options[MSTEP]:
        return

    y_ndim, ntrial, nbin, x_ndim = model['x'].shape  # neuron, trial, time, regression
    ntrial, nbin, z_ndim = model['mu'].shape
    y_types = model['obs_types']

    a = model['a']
    da = model['da']
    db = model['db']
    b = model['b']

    y = model['y'].reshape((-1, y_ndim))  # concatenate trials
    x = model['x'].reshape((y_ndim, -1, x_ndim))  # concatenate trials

    mu = model['mu'].reshape((-1, z_ndim))
    v = model['v'].reshape((-1, z_ndim))

    for i in range(options['m_niter']):
        eta = mu @ a + einsum('ijk, ki -> ji', x,
                              b)  # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
        r = sexp(eta + 0.5 * v @ (a ** 2))
        model['noise'] = var(y - eta, axis=0, ddof=0)  # MLE

        for y_dim in range(y_ndim):
            if y_types[y_dim] == SPIKE:
                # loading
                mu_plus_v_times_a = mu + v * a[:, y_dim]
                grad_a = mu.T @ y[:, y_dim] - mu_plus_v_times_a.T @ r[:, y_dim]

                if options['hessian']:
                    neghess_a = mu_plus_v_times_a.T @ (r[:, [y_dim]] * mu_plus_v_times_a)  # + wv
                    neghess_a[np.diag_indices_from(neghess_a)] += r[:, y_dim] @ v

                    try:
                        delta_a = solve(neghess_a, grad_a, sym_pos=True)
                    except LinAlgError:
                        delta_a = options['learning_rate'] * grad_a
                else:
                    delta_a = options['learning_rate'] * grad_a

                clip(delta_a, options['da_bound'])
                da[:, y_dim] = delta_a
                a[:, y_dim] += delta_a

                # regression
                grad_b = x[y_dim, :].T @ (y[:, y_dim] - r[:, y_dim])

                if options['hessian']:
                    neghess_b = x[y_dim, :].T @ (r[:, [y_dim]] * x[y_dim, :])
                    try:
                        delta_b = solve(neghess_b, grad_b, sym_pos=True)
                    except LinAlgError:
                        delta_b = options['learning_rate'] * grad_b
                else:
                    delta_b = options['learning_rate'] * grad_b

                clip(delta_b, options['db_bound'])
                db[:, y_dim] = delta_b
                b[:, y_dim] += delta_b
            elif y_types[y_dim] == LFP:
                # a's least squares solution for Gaussian channel
                # (m'm + diag(j'v))^-1 m'(y - Hb)
                tmp = mu.T @ mu
                tmp[np.diag_indices_from(tmp)] += sum(v, axis=0)
                a[:, y_dim] = solve(tmp, mu.T @ (y[:, y_dim] - x[y_dim, :] @ b[:, y_dim]), sym_pos=True)

                # b's least squares solution for Gaussian channel
                # (H'H)^-1 H'(y - ma)
                b[:, y_dim] = solve(x[y_dim, :].T @ x[y_dim, :],
                                    x[y_dim, :].T @ (y[:, y_dim] - mu @ a[:, y_dim]), sym_pos=True)
            else:
                pass

        # normalize loading by latent and rescale latent
        constrain_a(model)

        if norm(da) < options['tol'] * norm(a) and norm(db) < options['tol'] * norm(b):
            break


def hstep(model):
    """Optimize hyperparameters"""
    options = model['options']
    if not options[MSTEP]:
        return

    if model[ITER] % options[HPERIOD] != 0:
        return

    ntrial, nbin, z_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    mu = model['mu']
    w = model['w']
    subsample_size = options['subsample_size']
    if subsample_size is None:
        subsample_size = nbin // 2
    sigma = model['sigma']
    omega = model['omega']
    for z_dim in range(z_ndim):
        subsample = hyper.subsample(nbin, subsample_size)
        hparam_init = (sigma[z_dim] ** 2, omega[z_dim], options['gp_noise'])
        bounds = ((1e-3, 1),
                  options['omega_bound'],
                  (options['gp_noise'] / 2, options['gp_noise'] * 2))
        mask = np.array([0, 1, 0])
        sigma2, omega[z_dim], _ = hyper.optim(options[HOBJ],
                                              subsample,
                                              mu[:, subsample, z_dim].T,
                                              w[:, subsample, z_dim].T,
                                              hparam_init,
                                              bounds,
                                              mask=mask,
                                              return_f=False)  # noise variance, small value to avoid oversmoothing
        sigma[z_dim] = sqrt(sigma2)
    model['chol'] = np.array(
        [ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim] for dyn_dim in range(z_ndim)])


def em(model, callback=None):
    options = model['options']
    tol = options['tol']
    niter = options['niter']

    #######################
    # iterative algorithm #
    #######################
    gc.disable()  # disable gabbage collection during the iterative procedure

    try:
        for it in range(niter):
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

            callback(model)

            #####################
            # convergence check #
            #####################
            mu = model['mu']
            dmu = model['dmu']
            a = model['a']
            da = model['da']
            b = model['b']
            db = model['db']

            converged = norm(dmu) < tol * norm(mu) and norm(da) < tol * norm(a) and norm(db) < tol * norm(b)
            stop = converged

            if stop:
                break
    finally:
        util.save(model, model['path'])

    ##############################
    # end of iterative procedure #
    ##############################
    gc.enable()  # enable gabbage collection


def check_y_type(types):
    types = asarray(types)
    coded_types = np.empty_like(types, dtype=int)
    for i, type_ in enumerate(types):
        if type_ == 'spike':
            coded_types[i] = SPIKE
        elif type_ == 'lfp':
            coded_types[i] = LFP
        else:
            coded_types[i] = UNUSED
    return coded_types


def make_regression(x, y, history_filter=0):
    return None


def postprocess(model):
    """
    Remove intermediate and empty variables, and compute decomposition of posterior covariance.

    Parameters
    ----------
    model : dict
        raw fit

    Returns
    -------
    dict
        fit that contains prior, posterior, loading and regression
    """
    ntrial, nbin, z_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    w = model['w']
    eyer = identity(rank)
    L = empty((ntrial, z_ndim, nbin, rank))
    for trial in range(ntrial):
        for z_dim in range(z_ndim):
            G = prior[z_dim]
            GtWG = G.T @ (w[trial, :, [z_dim]].T * G)
            try:
                tmp = eyer - GtWG + GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)  # A should be PD but numerically not
            except LinAlgError:
                warnings.warn('Singular matrix. Use least squares instead.')
                tmp = eyer - GtWG + GtWG @ lstsq(eyer + GtWG, GtWG)[0]  # least squares
            eigval, eigvec = eigh(tmp)
            eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
            L[trial, z_dim, :] = G @ (eigvec @ diag(sqrt(eigval)))
    model['L'] = L
    model.pop('h')
    model.pop('stat')


def predict(z, a, b, y=None, v=None):
    """
    Predict firing rate

    Parameters
    ----------
    z : ndarray
        latent
    a : ndarray
        loading
    b : ndarray
        history filter
    y : ndarray
        spike trains for history filter
    v : ndarray
        posterior variance

    Returns
    -------
    ndarray
        predicted firing rate
    """
    ntrial, nbin, z_ndim = z.shape
    y_ndim = a.shape[1]
    history_filter = b.shape[0] - 1

    shape_out = (ntrial, nbin, y_ndim)
    # regression (h dot b) part
    if y is None:
        y = np.zeros(shape_out)

    hb = empty(shape_out)
    for y_dim in range(y_ndim):
        for trial in range(ntrial):
            h = add_constant(lagmat(y[trial, :, y_dim], lag=history_filter))
            hb[trial, :, y_dim] = h @ b[:, y_dim]
    eta = z.reshape((-1, z_ndim)) @ a + hb.reshape((-1, y_ndim))
    r = sexp(eta + 0.5 * v.reshape((-1, z_ndim)) @ (a ** 2)) if v is not None else sexp(eta)
    return np.reshape(r, shape_out)


def clip(delta, lbound, ubound=None):
    if ubound is None:
        assert (lbound > 0)
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(delta, lbound, ubound, out=delta)


def update_w(model):
    obs_ndim, ntrial, nbin, nreg = model['x'].shape
    dyn_ndim = model['mu'].shape[-1]

    spike_dims = model['obs_types'] == SPIKE
    lfp_dims = model['obs_types'] == LFP

    flat_mu = model['mu'].reshape((-1, dyn_ndim))
    flat_x = model['x'].reshape((obs_ndim, -1, nreg))  # concatenate trials
    flat_v = model['v'].reshape((-1, dyn_ndim))
    shape_w = model['w'].shape

    eta = flat_mu @ model['a'] + einsum('ijk, ki -> ji', flat_x, model[
        'b'])  # (neuron, time, regression) x (regression, neuron) -> (time, neuron)
    r = sexp(eta + 0.5 * flat_v @ (model['a'] ** 2))
    U = empty_like(r)

    U[:, spike_dims] = r[:, spike_dims]
    U[:, lfp_dims] = 1 / model['noise'][lfp_dims]
    model['w'] = reshape(U @ (model['a'].T ** 2), shape_w)


def update_v(model):
    if model['options']['method'] == VB:
        prior = model['chol']
        rank = prior[0].shape[-1]
        Ir = identity(rank)
        ntrial, nbin, z_ndim = model['mu'].shape

        for trial in range(ntrial):
            w = model['w'][trial, :]
            for z_dim in range(z_ndim):
                G = prior[z_dim]
                GtWG = G.T @ (w[:, [z_dim]] * G)
                try:
                    model['v'][trial, :, z_dim] = (
                        G * (G - G @ GtWG + G @ (GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)))).sum(axis=1)
                except LinAlgError:
                    warnings.warn("singular I + G'WG")


def constrain_mu(model):
    options = model['options']
    if not options['constrain_mu']:
        return

    z_ndim = model['dyn_ndim']
    shape = model['mu'].shape
    mu_ = model['mu'].reshape((-1, z_ndim))
    mean_over_trials = mu_.mean(axis=0, keepdims=True)
    model['b'][0, :] += mean_over_trials @ model['a']  # compensate bias
    mu_ -= mean_over_trials
    model['mu'] = mu_.reshape(shape)


def constrain_a(model):
    options = model['options']
    if not options['constrain_a']:
        return

    method = options['constrain_a']
    eps = options['eps']

    shape_mu = model['mu'].shape
    mu_ = model['mu'].reshape((-1, shape_mu.shape[-1]))
    a = model['a']
    if method == 'none':
        return
    if method == 'svd':
        # SVD is not good as above
        # noinspection PyTupleAssignmentBalance
        U, s, Vh = svd(a, full_matrices=False)
        model['mu'] = np.reshape(mu_ @ a @ Vh.T, shape_mu)
        model['a'] = Vh
    else:
        s = norm(a, ord=method, axis=1, keepdims=True) + eps
        a /= s
        mu_ *= s.squeeze()  # compensate latent
        model['mu'] = mu_.reshape(shape_mu)
