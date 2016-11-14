import gc
import time
import warnings
from functools import partial
from pprint import pprint

import numpy as np
from numpy import identity, einsum, trace, empty, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, sum
from numpy.core.umath import sqrt, PINF, log
from numpy.linalg import slogdet
from scipy.linalg import lstsq, eigh, solve, norm, LinAlgError
from scipy.stats import spearmanr
from sklearn.decomposition import factor_analysis
from tqdm import tqdm

from vlgp import hyper
from vlgp import util
from vlgp.callback import Saver, Progress
from vlgp.experimental import hstep, estep, mstep, update_w, update_v, em
from .constant import *
from .evaluation import timer
from .math import ichol_gauss, subspace, sexp
from .util import add_constant, rotate, lagmat, save


def elbo(model_fit):
    """
    Evidence Lower BOund (ELBO)

    Parameters
    ----------
    model_fit : dict

    Returns
    -------
    lb : double
        lower bound
    ll : double
        log likelihood
    """
    obs_ndim, ntrial, nbin, lag1 = model_fit['h'].shape  # neuron, trial, time, lag1
    dyn_ndim = model_fit['mu'].shape[-1]
    prior = model_fit['chol']
    rank = prior[0].shape[-1]

    Ir = identity(rank)

    y = model_fit['y'].reshape((-1, obs_ndim))  # concatenate trials
    h = model_fit['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials
    obs_types = model_fit['channel']

    prior = model_fit['chol']

    mu = model_fit['mu'].reshape((-1, dyn_ndim))
    v = model_fit['v'].reshape((-1, dyn_ndim))

    a = model_fit['a']
    b = model_fit['b']
    noise = model_fit['noise']

    spike_dims = obs_types == SPIKE
    lfp_dims = obs_types == LFP

    eta = mu @ a + einsum('ijk, ki -> ji', h.reshape((obs_ndim, nbin * ntrial, lag1)), b)
    r = sexp(eta + 0.5 * v @ (a ** 2))
    # possible useless calculation here and for noise when spike and LFP mixed.
    # LFP (Gaussian) has no firing rate and spike (Poisson) has no 'noise'.
    # useless dims could be removed to save computational time and space.

    llspike = sum(y[:, spike_dims] * eta[:, spike_dims] - r[:, spike_dims])  # verified by predict()

    # noinspection PyTypeChecker
    lllfp = - 0.5 * sum(((y[:, lfp_dims] - eta[:, lfp_dims]) ** 2 + v @ (a[:, lfp_dims] ** 2)) / noise[lfp_dims] + log(
        noise[lfp_dims] + log(2 * np.pi)))

    ll = llspike + lllfp

    lb = ll

    eps = 1e-3

    for trial in range(ntrial):
        mu = model_fit['mu'][trial, :]
        w = model_fit['w'][trial, :]
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


def infer(model, options):
    """
    Inference procedure

    Parameters
    ----------
    model : dict
    options : dict

    Returns
    -------
    dict
        model fit
    """

    # def estep():
    #     """Optimize posterior (E step)"""
    #     obs_ndim = model['y'].shape[-1]
    #     ntrial, nbin, dyn_ndim = model['mu'].shape
    #     prior = model['chol']
    #     rank = prior[0].shape[-1]
    #     a = model['a']
    #     b = model['b']
    #     noise = model['noise']
    #
    #     eyer = identity(rank)
    #     residual = empty((nbin, obs_ndim), dtype=float)
    #     U = empty((nbin, obs_ndim), dtype=float)
    #
    #     for trial in range(ntrial):
    #         # trial slices
    #         y = model['y'][trial, :]
    #         h = model['h'][:, trial, :, :]
    #         mu = model['mu'][trial, :]
    #         w = model['w'][trial, :]
    #         v = model['v'][trial, :]
    #
    #         hb = einsum('ijk, ki -> ji', h, b)
    #         eta = mu @ a + hb
    #         r = sexp(eta + 0.5 * v @ (a ** 2))
    #         for dyn_dim in range(dyn_ndim):
    #             G = prior[dyn_dim]
    #
    #             # working residuals
    #             # extensible to many other distributions
    #             # very similar form to GLM
    #             residual[:, spike_dims] = y[:, spike_dims] - r[:, spike_dims]
    #             residual[:, lfp_dims] = (y[:, lfp_dims] - eta[:, lfp_dims]) / noise[lfp_dims]
    #
    #             # if adjust_hessian:
    #             #     grad_mu_resid = (y[:, spike_dims] - r[:, spike_dims]) @ a[dyn_dim, spike_dims] + \
    #             #                 ((y[:, lfp_dims] - eta[:, lfp_dims]) / noise[lfp_dims]) @ a[dyn_dim, lfp_dims]
    #             #     grad_mu = grad_mu_resid - lstsq(G.T, lstsq(G, mu[:, dyn_dim])[0])[0]
    #             #     dmu_acc[trial, :, dyn_dim] = accumulate(dmu_acc[trial, :, dyn_dim], grad_mu, decay)
    #             #     wadj = w[:, dyn_dim] + eps + sqrt(dmu_acc[trial, :, dyn_dim])  # adjusted Hessian
    #             # else:
    #             wadj = w[:, [dyn_dim]]  # keep dimension
    #             # wadj = wadj[:, None]
    #             GtWG = G.T @ (wadj * G)
    #
    #             u = G @ (G.T @ (residual @ a[dyn_dim, :])) - mu[:, dyn_dim]
    #             try:
    #                 delta_mu = u - G @ ((wadj * G).T @ u) + \
    #                            G @ (GtWG @ solve(eyer + GtWG, (wadj * G).T @ u, sym_pos=True))
    #             except LinAlgError:
    #                 delta_mu = 0
    #
    #             clip(delta_mu, options['dmu_bound'])
    #             # if options['Adam']:
    #             #     optimizer = options['optimizer_mu'][trial, dyn_dim]
    #             #     delta_mu = optimizer.next_update(delta_mu)
    #             mu[:, dyn_dim] += options['learning_rate'] * delta_mu
    #
    #         eta = mu @ a + hb
    #         r = sexp(eta + 0.5 * v @ (a ** 2))
    #         U[:, spike_dims] = r[:, spike_dims]
    #         U[:, lfp_dims] = 1 / noise[lfp_dims]
    #         copyto(w, U @ (a.T ** 2))
    #         if options['method'] == 'VB':
    #             for dyn_dim in range(dyn_ndim):
    #                 G = prior[dyn_dim]
    #                 GtWG = G.T @ (w[:, [dyn_dim]] * G)
    #                 try:
    #                     v[:, dyn_dim] = (G * (G - G @ GtWG + G @ (GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)))).sum(
    #                         axis=1)
    #                 except LinAlgError:
    #                     warnings.warn("singular I + G'WG")
    #
    #     # center over all trials if not only infer posterior
    #     if options['constrain_mu']:
    #         shape = model['mu'].shape
    #         mu_over_trials = model['mu'].reshape((-1, dyn_ndim))
    #         mean_over_trials = mu_over_trials.mean(axis=0)
    #         model['b'][0, :] += mean_over_trials @ model['a']  # compensate bias
    #         mu_over_trials -= mean_over_trials
    #         model['mu'] = mu_over_trials.reshape(shape)

    # def mstep():
    #     """Optimize loading and regression (M step)"""
    #     y_ndim, ntrial, nbin, lag1 = model['h'].shape  # neuron, trial, time, lag
    #     ntrial, nbin, dyn_ndim = model['mu'].shape
    #     obs_types = model['channel']
    #     a = model['a']
    #     b = model['b']
    #
    #     y = model['y'].reshape((-1, y_ndim))  # concatenate trials
    #     h = model['h'].reshape((y_ndim, -1, lag1))  # concatenate trials
    #
    #     mu = model['mu'].reshape((-1, dyn_ndim))
    #     v = model['v'].reshape((-1, dyn_ndim))
    #
    #     eta = mu @ a + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) x (lag, neuron) -> (time, neuron)
    #     r = sexp(eta + 0.5 * v @ (a ** 2))
    #     model['noise'] = var(y - eta, axis=0, ddof=0)  # MLE
    #
    #     for y_dim in range(y_ndim):
    #         if obs_types[y_dim] == SPIKE:
    #             # loading
    #             mu_plus_v_times_a = mu + v * a[:, y_dim]
    #             grad_a = mu.T @ y[:, y_dim] - mu_plus_v_times_a.T @ r[:, y_dim]
    #
    #             if options['hessian']:
    #                 neghess_a = mu_plus_v_times_a.T @ (r[:, [y_dim]] * mu_plus_v_times_a)  # + wv
    #                 neghess_a[np.diag_indices_from(neghess_a)] += r[:, y_dim] @ v
    #
    #                 try:
    #                     delta_a = solve(neghess_a, grad_a, sym_pos=True)
    #                 except LinAlgError:
    #                     delta_a = grad_a
    #                 except ValueError:
    #                     delta_a = grad_a
    #             else:
    #                 delta_a = grad_a
    #
    #             clip(delta_a, options['da_bound'])
    #             # if options['Adam']:
    #             #     optimizer_a = options['optimizer_a'][obs_dim]
    #             #     delta_a = optimizer_a.next_update(delta_a)
    #             a[:, y_dim] += options['learning_rate'] * delta_a
    #
    #             # regression
    #             grad_b = h[y_dim, :].T @ (y[:, y_dim] - r[:, y_dim])
    #
    #             if options['hessian']:
    #                 neghess_b = h[y_dim, :].T @ (r[:, [y_dim]] * h[y_dim, :])
    #                 # TODO: inactive neurons never fire across all trials which may cause zero Hessian
    #                 # if adjust_hessian:
    #                 # db_acc[:, obs_dim] = accumulate(db_acc[:, obs_dim], grad_b, decay)
    #                 # neghess_b[np.diag_indices_from(neghess_b)] += eps + sqrt(db_acc[:, obs_dim])
    #                 try:
    #                     delta_b = solve(neghess_b, grad_b, sym_pos=True)
    #                 except LinAlgError:
    #                     # print('singular Hessian b')
    #                     delta_b = grad_b
    #             else:
    #                 delta_b = grad_b
    #
    #             clip(delta_b, options['db_bound'])
    #             # if options['Adam']:
    #             #     optimizer_b = options['optimizer_b'][obs_dim]
    #             #     delta_b = optimizer_b.next_update(delta_b)
    #             b[:, y_dim] += options['learning_rate'] * delta_b
    #         elif obs_types[y_dim] == LFP:
    #             # a's least squares solution for Gaussian channel
    #             # (m'm + diag(j'v))^-1 m'(y - Hb)
    #             tmp = mu.T @ mu
    #             tmp[np.diag_indices_from(tmp)] += sum(v, axis=0)
    #             a[:, y_dim] = solve(tmp, mu.T @ (y[:, y_dim] - h[y_dim, :] @ b[:, y_dim]), sym_pos=True)
    #
    #             # b's least squares solution for Gaussian channel
    #             # (H'H)^-1 H'(y - ma)
    #             # b[:, y_dim] = solve(h[y_dim, :].T @ h[y_dim, :],
    #             #                     h[y_dim, :].T @ (y[:, y_dim] - mu @ a[:, y_dim]), sym_pos=True)
    #             b[:, y_dim] = solve(h[y_dim, :].T @ h[y_dim, :],
    #                                 h[y_dim, :].T @ (y[:, y_dim] - mu @ a[:, y_dim]), sym_pos=True)
    #             b[1:, y_dim] = 0
    #         else:
    #             pass
    #
    #     # normalize loading by latent and rescale latent
    #     constrain_loading(model, method=options['constrain_a'], eps=options['eps'])

    # def hstep():
    #     """Optimize hyperparameters"""
    #     ntrial, nbin, z_ndim = model['mu'].shape
    #     prior = model['chol']
    #     rank = prior[0].shape[-1]
    #     mu = model['mu']
    #     w = model['w']
    #     subsample_size = options['subsample_size']
    #     if subsample_size is None:
    #         subsample_size = nbin // 2
    #     sigma = model['sigma']
    #     omega = model['omega']
    #     for z_dim in range(z_ndim):
    #         subsample = hyper.subsample(nbin, subsample_size)
    #         init_p = (sigma[z_dim] ** 2, omega[z_dim], options['gp_noise'])
    #         bounds = ((1e-3, 1),
    #                   options['omega_bound'],
    #                   (options['gp_noise'] / 2, options['gp_noise'] * 2))
    #         mask = np.array([0, 1, 0])
    #         sigma2, omega_new, _ = hyper.optim(options['hyper_obj'],
    #                                               subsample,
    #                                               mu[:, subsample, z_dim].T,
    #                                               w[:, subsample, z_dim].T,
    #                                               init_p,
    #                                               bounds,
    #                                               mask=mask,
    #                                               return_f=False)  # noise variance, small value to avoid oversmoothing
    #         if not np.any(np.isclose(omega_new, options['omega_bound'])):
    #             # unattainable bounds
    #             omega[z_dim] = omega_new
    #         sigma[z_dim] = sqrt(sigma2)
    #     copyto(good_sigma, sigma)
    #     copyto(good_omega, omega)
    #     model['chol'] = np.array(
    #         [ichol_gauss(nbin, omega[z_dim], rank) * sigma[z_dim] for z_dim in range(z_ndim)])

    #################
    # function body #
    #################

    # truth
    x = model.get('x')
    alpha = model.get('alpha')

    # options
    # eps = options['eps']
    tol = options['tol']

    # spike_dims = model['channel'] == SPIKE
    # lfp_dims = model['channel'] == LFP

    ################################
    # old values
    # good_mu = model['mu'].copy()
    # good_w = model['w'].copy()
    # good_v = model['v'].copy()
    # good_a = model['a'].copy()
    # good_b = model['b'].copy()
    # good_noise = model['noise'].copy()
    # good_sigma = model['sigma'].copy()
    # good_omega = model['omega'].copy()

    stat = empty(options['niter'], dtype=object)
    lb = zeros(options['niter'], dtype=float)
    ll = zeros(options['niter'], dtype=float)
    elapsed = zeros((options['niter'], 3), dtype=float)
    loading_angle = zeros(options['niter'], dtype=float)
    latent_angle = zeros(options['niter'], dtype=float)
    latent_corr = zeros((options['niter'], model['mu'].shape[-1]), dtype=float)

    # iteration 0
    # lb[0], ll[0] = elbo(obj)
    lb[0], ll[0] = np.finfo(float).min, np.finfo(float).min
    if alpha is not None:
        loading_angle[0] = subspace(alpha.T, model['a'].T)
    if x is not None:
        rotated = empty_like(x, dtype=float)
        # rotate trial by trial
        for itrial in range(x.shape[0]):
            rotated[itrial, :] = rotate(add_constant(model['mu'][itrial, :]), x[itrial, :])
        latent_angle[0] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
        rho, _ = spearmanr(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
        latent_corr[0] = rho[np.arange(x.shape[-1]), np.arange(x.shape[-1]) + x.shape[-1]]

    # iterative algorithm
    it = 1  # iteration counter
    stop = False

    logging_counter = 0
    last_saving_time = time.perf_counter()

    if options['verbose']:
        print('\nstarting')

    #######################
    # iterative algorithm #
    #######################
    gc.disable()  # disable gabbage collection during the iterative procedure
    try:
        with tqdm(total=options['niter'] - 1) as pbar:
            while not stop and it < options['niter']:
                with timer() as iter_elapsed:
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
                # anneal learning rate
                # options['learning_rate'] = 1 / (1 + it / options['niter'])

                # Calculate angle between latent subspace if true latent is given.
                if x is not None:
                    for itrial in range(x.shape[0]):
                        rotated[itrial, :] = rotate(add_constant(model['mu'][itrial, :]), x[itrial, :])
                    latent_angle[it] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
                    rho, _ = spearmanr(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
                    latent_corr[it] = rho[np.arange(x.shape[-1]), np.arange(x.shape[-1]) + x.shape[-1]]

                # Calculate angle between loading subspace if true loading is given.
                if alpha is not None:
                    loading_angle[it] = subspace(alpha.T, model['a'].T)

                #####################
                # convergence check #
                #####################
                # lb[it], ll[it] = 0, 0
                # converged = norm(model['mu'].ravel() - good_mu.ravel()) <= (
                #     eps + tol * norm(good_mu.ravel())) and norm(
                #     model['a'].ravel() - good_a.ravel()) <= (eps + tol * norm(good_a.ravel())) and norm(
                #     model['b'].ravel() - good_b.ravel()) <= (eps + tol * norm(good_b.ravel()))
                #
                # copyto(good_mu, model['mu'])
                # copyto(good_w, model['w'])
                # copyto(good_v, model['v'])
                # copyto(good_a, model['a'])
                # copyto(good_b, model['b'])
                # copyto(good_noise, model['noise'])
                mu = model['mu']
                dmu = model['dmu']
                a = model['a']
                da = model['da']
                b = model['b']
                db = model['db']

                converged = norm(dmu) < tol * norm(mu) and norm(da) < tol * norm(a) and norm(db) < tol * norm(b)
                stop = converged

                elapsed[it, 0] = estep_elapsed()
                elapsed[it, 1] = mstep_elapsed()
                elapsed[it, 2] = iter_elapsed()

                ###################################
                # statistics of current iteration #
                ###################################

                stat[it] = dict()
                stat[it]['E-step elapsed'] = elapsed[it, 0]
                stat[it]['M-step elapsed'] = elapsed[it, 1]
                stat[it]['H-step elapsed'] = hstep_elapsed()
                stat[it]['Total elapsed'] = elapsed[it, 2]
                # stat[it]['ELBO'] = lb[it]
                # stat[it]['LL'] = ll[it]
                stat[it]['sigma'] = model['sigma']
                stat[it]['omega'] = model['oemga']

                if options['verbose']:  # and it == 2 ** logging_counter:
                    print('\n[{}]'.format(it))
                    pprint(stat[it])
                    logging_counter += 1

                it += 1
                pbar.update(1)

                model['it'] = it
                model['ELBO'] = lb[:it]
                model['Elapsed'] = elapsed[:it, :]
                model['LoadingAngle'] = loading_angle[:it]
                model['LatentAngle'] = latent_angle[:it]
                model['RankCorr'] = latent_corr[:it]
                model['LL'] = ll[:it]
                model['stat'] = stat

                now = time.perf_counter()
                if now - last_saving_time > options['saving_interval']:
                    print('Saving')
                    save(model, model['path'])
                    last_saving_time = now
                    print('Saved')
    except KeyboardInterrupt:
        print('Quit')
    finally:
        print('Saving')
        save(model, model['path'])
        print('Saved')

    ##############################
    # end of iterative procedure #
    ##############################
    gc.enable()  # enable gabbage collection

    lb[it - 1], ll[it - 1] = elbo(model)
    if options['verbose']:
        print('\nInference ends')
        print('{} iterations, ELBO: {:.4f}\n'.format(it - 1, lb[it - 1]))

    # model['ELBO'] = lb[:it]
    # model['Elapsed'] = elapsed[:it, :]
    # model['LoadingAngle'] = loading_angle[:it]
    # model['LatentAngle'] = latent_angle[:it]
    # model['RankCorr'] = latent_corr[:it]
    # model['LL'] = ll[:it]
    # model['stat'] = stat
    return model


def check_obs_type(types):
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


def fit(y,
        obs_types,
        dyn_ndim,
        exog=None,
        a=None,
        b=None,
        mu=None,
        lag=0,
        x=None,
        alpha=None,
        beta=None,
        sigma=None,
        omega=None,
        rank=None,
        path='vlgpfit',
        callbacks=None,
        **kwargs):
    """
    vLGP main function

    Parameters
    ----------
    y : ndarray
        obserbation
    obs_types : ndarray
        types of observation dimensions, 'spike' or 'lfp'
    dyn_ndim : int
        number of latent dimensions
    exog : ndarray
        external factors
    a : ndarray, optional
        initial value of loading
    b : ndarray, optional
        initial value of regression
    mu : ndarray, optional
        initial value of posterior mean
    x : ndarray, optional
        true value of latent
    alpha : ndarray, optional
        true value of loading
    beta : ndarray, optional
        true value of regression
    lag : int, optional
        autoregressive lag
    rank : int, optional
        rank of incomplete Cholesky
    eps : double, optional
        a small positive number
    tol : double, optional
        numerical tolerance
    path : string, optional
        path to the save file
    kwargs : dict, optional
        algorithm options. See fill_options()

    Returns
    -------
    dict
        fit
    """
    options = check_options(kwargs)

    callbacks = callbacks or []
    # tol = options['tol']
    eps = options['eps']

    y = asarray(y)
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]

    obs_types = check_obs_type(obs_types)

    ntrial, nbin, obs_ndim = y.shape

    # make design matrix of regression
    h = empty((obs_ndim, ntrial, nbin, 1 + lag), dtype=float)
    for obs_dim in range(obs_ndim):
        for trial in range(ntrial):
            h[obs_dim, trial, :] = add_constant(lagmat(y[trial, :, obs_dim], lag=lag))

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent
    if a is None and mu is None:
        fa = factor_analysis.FactorAnalysis(n_components=dyn_ndim, svd_method='lapack')
        y_ = y.reshape((-1, obs_ndim))
        y0 = y[0, :]
        fa.fit(y0)
        a = fa.components_
        mu = fa.transform(y_)

        # constrain loading and center latent
        scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
        a /= scale
        mu *= scale.squeeze()  # compensate latent
        mu -= mu.mean(axis=0)
        mu = mu.reshape((ntrial, nbin, dyn_ndim))

        # noinspection PyTupleAssignmentBalance
        # U, s, Vh = svd(a, full_matrices=False)
        # mu = np.reshape(mu @ a @ Vh.T, (ntrial, nbin, nlatent))
        # a[:] = Vh
    else:
        if mu is None:
            mu = lstsq(a.T, y.reshape((-1, obs_ndim)).T)[0].T.reshape((ntrial, nbin, dyn_ndim))
        elif a is None:
            a = lstsq(mu.reshape((-1, dyn_ndim)), y.reshape((-1, obs_ndim)))[0]

    # initialize regression
    spike_dims = obs_types == SPIKE

    if b is None:
        b = empty((1 + lag, obs_ndim), dtype=float)
        for obs_dim in np.arange(obs_ndim)[spike_dims]:
            b[:, obs_dim] = \
                lstsq(h.reshape((obs_ndim, -1, 1 + lag))[obs_dim, :], y.reshape((-1, obs_ndim))[:, obs_dim])[0]

    # initialize noises of LFP
    noise = var(y.reshape((-1, obs_ndim)), axis=0, ddof=0)

    ####################
    # initialize prior #
    ####################
    if sigma is None:
        sigma = np.ones(dyn_ndim) * (1 - options['gp_noise'])

    if omega is None:
        if options['subsample_size'] is None:
            options['subsample_size'] = nbin // 5
        subsample_size = options['subsample_size']
        omega = np.ones(dyn_ndim)
        for dyn_dim in range(dyn_ndim):
            subsample = hyper.subsample(nbin, subsample_size, kwargs['successive'])
            omega_grid = np.logspace(-6, 0, num=7, base=10)
            sigma2_opt = np.zeros_like(omega_grid)
            omega_opt = np.zeros_like(omega_grid)
            fval_opt = np.zeros_like(omega_grid)

            bounds = ((1e-3, 1.0),
                      options['omega_bound'],
                      (options['gp_noise'] / 2, options['gp_noise'] * 2))
            mask = np.array([0, 1, 0])  # only optimize omega
            for i, o in enumerate(omega_grid):
                init_p = (1 - options['gp_noise'], o, options['gp_noise'])
                (sigma2_opt[i], omega_opt[i], _), fval_opt[i] = hyper.optim('GP',
                                                                            subsample,  # time
                                                                            mu[:, subsample, dyn_dim].T,
                                                                            None,  # Sigma
                                                                            init_p,
                                                                            bounds=bounds,
                                                                            mask=mask,
                                                                            return_f=True)
            best = np.argmin(fval_opt)
            sigma[dyn_dim] = sqrt(sigma2_opt[best])
            omega[dyn_dim] = omega_opt[best]

    # make Cholesky of prior
    if rank is None:
        rank = nbin

    prior = np.array([ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim] for dyn_dim in range(dyn_ndim)])

    # w and v
    w = zeros_like(mu, dtype=float)
    v = zeros_like(mu, dtype=float)

    model = dict(y=y,
                 channel=obs_types,
                 dyn_ndim=dyn_ndim,
                 h=h,
                 sigma=sigma,
                 omega=omega,
                 chol=prior,
                 mu=mu,
                 w=w,
                 v=v,
                 L=empty((ntrial, dyn_ndim, nbin, rank)),
                 a=a,
                 b=b,
                 noise=noise,
                 x=x,
                 alpha=alpha,
                 beta=beta,
                 path=path,
                 options=options)

    update_w(model)
    if options['method'] == 'VB':
        update_v(model)

    saver = Saver(model)
    pbar = Progress(model)

    callbacks.extend([pbar, saver])
    try:
        em(model, callbacks)
    finally:
        util.save(model, model['path'])

    return model


def check_options(kwargs):
    """
    Fill missing options with default values

    Parameters
    ----------
    kwargs : dict
        options with missing values

    Returns
    -------
    dict
        full options
    """
    options = dict(kwargs)
    for k, v in DEFAULT_OPTIONS.items():
        # If key is in the dictionary, return its value. If not, insert key with a value of default and return default.
        options.setdefault(k, v)
    return options


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
    ntrial, nbin, dyn_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    w = model['w']
    eyer = identity(rank)
    L = empty((ntrial, dyn_ndim, nbin, rank))
    for trial in range(ntrial):
        for dyn_dim in range(dyn_ndim):
            G = prior[dyn_dim]
            GtWG = G.T @ (w[trial, :, [dyn_dim]].T * G)
            try:
                tmp = eyer - GtWG + GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)  # A should be PD but numerically not
            except LinAlgError:
                warnings.warn('Singular matrix. Use least squares instead.')
                tmp = eyer - GtWG + GtWG @ lstsq(eyer + GtWG, GtWG)[0]  # least squares
            eigval, eigvec = eigh(tmp)
            eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
            L[trial, dyn_dim, :] = G @ (eigvec @ diag(sqrt(eigval)))
    model['L'] = L
    keys = list(model.keys())
    for key in keys:
        if model.get(key, None) is None:
            model.pop(key, None)
    model.pop('h', None)
    model.pop('stat', None)
    return model


def predict(x, a, b, y=None, v=None):
    """
    Predict firing rate

    Parameters
    ----------
    x : ndarray
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
    ntrial, nbin, dyn_ndim = x.shape
    obs_ndim = a.shape[1]
    lag = b.shape[0] - 1

    shape_out = (ntrial, nbin, obs_ndim)
    # regression (h dot b) part
    if y is None:
        y = np.zeros(shape_out)

    h_dot_b = empty(shape_out)
    for obs_dim in range(obs_ndim):
        for trial in range(ntrial):
            h = add_constant(lagmat(y[trial, :, obs_dim], lag=lag))
            h_dot_b[trial, :, obs_dim] = h @ b[:, obs_dim]
    eta = x.reshape((-1, dyn_ndim)) @ a + h_dot_b.reshape((-1, obs_ndim))
    yhat = np.exp(eta + 0.5 * v.reshape((-1, dyn_ndim)) @ (a ** 2)) if v is not None else np.exp(eta)
    return np.reshape(yhat, shape_out)


def clip(delta, lbound, ubound=None):
    if ubound is None:
        assert (lbound > 0)
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(delta, lbound, ubound, out=delta)

# def constrain_loading(model_fit, method=inf, eps=1e-8):
#     mu = model_fit['mu']
#     shape_mu = mu.shape
#     mu = mu.reshape((-1, mu.shape[-1]))
#     a = model_fit['a']
#     if method == 'none':
#         return
#     if method == 'svd':
#         # SVD is not good as above
#         # noinspection PyTupleAssignmentBalance
#         U, s, Vh = svd(a, full_matrices=False)
#         model_fit['mu'] = np.reshape(mu @ a @ Vh.T, shape_mu)
#         model_fit['a'] = Vh
#     else:
#         s = norm(a, ord=method, axis=1, keepdims=True) + eps
#         a /= s
#         mu *= s.squeeze()  # compensate latent
#         model_fit['mu'] = mu.reshape(shape_mu)
#
#
# def update_w(model_fit):
#     obs_ndim, ntrial, nbin, lag1 = model_fit['h'].shape
#     dyn_ndim = model_fit['mu'].shape[-1]
#
#     spike_dims = model_fit['channel'] == SPIKE
#     lfp_dims = model_fit['channel'] == LFP
#
#     flat_mu = model_fit['mu'].reshape((-1, dyn_ndim))
#     flat_h = model_fit['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials
#     flat_v = model_fit['v'].reshape((-1, dyn_ndim))
#     shape_w = model_fit['w'].shape
#
#     eta = flat_mu @ model_fit['a'] + einsum('ijk, ki -> ji', flat_h,
#                                             model_fit['b'])  # (neuron, time, lag) x (lag, neuron) -> (time, neuron)
#     r = sexp(eta + 0.5 * flat_v @ (model_fit['a'] ** 2))
#     U = empty_like(r)
#
#     U[:, spike_dims] = r[:, spike_dims]
#     U[:, lfp_dims] = 1 / model_fit['noise'][lfp_dims]
#     model_fit['w'] = reshape(U @ (model_fit['a'].T ** 2), shape_w)
#
#
# def update_v(model_fit):
#     prior = model_fit['chol']
#     rank = prior[0].shape[-1]
#     eyer = identity(rank)
#     ntrial, nbin, dyn_ndim = model_fit['mu'].shape
#
#     for trial in range(ntrial):
#         w = model_fit['w'][trial, :]
#         for dyn_dim in range(dyn_ndim):
#             G = prior[dyn_dim]
#             GtWG = G.T @ (w[:, [dyn_dim]] * G)
#             try:
#                 model_fit['v'][trial, :, dyn_dim] = (
#                     G * (G - G @ GtWG + G @ (GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)))).sum(axis=1)
#             except LinAlgError:
#                 warnings.warn("singular I + G'WG")
