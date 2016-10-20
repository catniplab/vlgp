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
from tqdm import tqdm
from tqdm import trange

from vlgp import hyper
from .constant import *
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
    obs_ndim, ntrial, nbin, lag1 = model['h'].shape  # neuron, trial, time, lag1
    dyn_ndim = model['mu'].shape[-1]
    prior = model['chol']
    rank = prior[0].shape[-1]

    Ir = identity(rank)

    y = model['y'].reshape((-1, obs_ndim))  # concatenate trials
    h = model['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials
    obs_types = model['channel']

    prior = model['chol']

    mu = model['mu'].reshape((-1, dyn_ndim))
    v = model['v'].reshape((-1, dyn_ndim))

    a = model['a']
    b = model['b']
    noise = model['noise']

    spike_dims = obs_types == SPIKE
    lfp_dims = obs_types == LFP

    eta = mu @ a + einsum('ijk, ki -> ji', h.reshape((obs_ndim, nbin * ntrial, lag1)), b)
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


def estep(model):
    """Update variational distribution q (E step)"""
    obs_ndim = model['y'].shape[-1]
    ntrial, nbin, dyn_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    a = model['a']
    b = model['b']
    noise = model['noise']
    spike_dims = model['channel'] == SPIKE
    lfp_dims = model['channel'] == LFP
    options = model['options']

    Ir = identity(rank)
    residual = empty((nbin, obs_ndim), dtype=float)
    U = empty((nbin, obs_ndim), dtype=float)

    for trial in range(ntrial):
        # trial slices
        y = model['y'][trial, :]
        h = model['h'][:, trial, :, :]
        mu = model['mu'][trial, :]
        w = model['w'][trial, :]
        v = model['v'][trial, :]

        hb = einsum('ijk, ki -> ji', h, b)
        eta = mu @ a + hb
        r = sexp(eta + 0.5 * v @ (a ** 2))
        for dyn_dim in range(dyn_ndim):
            G = prior[dyn_dim]

            # working residuals
            # extensible to many other distributions
            # similar form to GLM
            residual[:, spike_dims] = y[:, spike_dims] - r[:, spike_dims]
            residual[:, lfp_dims] = (y[:, lfp_dims] - eta[:, lfp_dims]) / noise[lfp_dims]

            wadj = w[:, [dyn_dim]]  # keep dimension
            GtWG = G.T @ (wadj * G)

            u = G @ (G.T @ (residual @ a[dyn_dim, :])) - mu[:, dyn_dim]
            delta_mu = u - G @ ((wadj * G).T @ u) + \
                       G @ (GtWG @ solve(Ir + GtWG, (wadj * G).T @ u, sym_pos=True))

            clip(delta_mu, options['dmu_bound'])
            mu[:, dyn_dim] += options['learning_rate'] * delta_mu

        eta = mu @ a + hb
        r = sexp(eta + 0.5 * v @ (a ** 2))
        U[:, spike_dims] = r[:, spike_dims]
        U[:, lfp_dims] = 1 / noise[lfp_dims]
        copyto(w, U @ (a.T ** 2))
        if options['method'] == 'VB':
            for dyn_dim in range(dyn_ndim):
                G = prior[dyn_dim]
                GtWG = G.T @ (w[:, [dyn_dim]] * G)
                try:
                    v[:, dyn_dim] = (G * (G - G @ GtWG + G @ (GtWG @ solve(Ir + GtWG, GtWG, sym_pos=True)))).sum(
                        axis=1)
                except LinAlgError:
                    warnings.warn("singular I + G'WG")

    # center over all trials if not only infer posterior
    if options['constrain_mu']:
        shape = model['mu'].shape
        mu_over_trials = model['mu'].reshape((-1, dyn_ndim))
        mean_over_trials = mu_over_trials.mean(axis=0)
        model['b'][0, :] += mean_over_trials @ model['a']  # compensate bias
        mu_over_trials -= mean_over_trials
        model['mu'] = mu_over_trials.reshape(shape)


def mstep(model):
    """Optimize loading and regression (M step)"""
    obs_ndim, ntrial, nbin, lag1 = model['h'].shape  # neuron, trial, time, lag
    ntrial, nbin, dyn_ndim = model['mu'].shape
    obs_types = model['channel']

    spike_dims = model['channel'] == SPIKE
    lfp_dims = model['channel'] == LFP
    options = model['options']

    a = model['a']
    b = model['b']

    y = model['y'].reshape((-1, obs_ndim))  # concatenate trials
    h = model['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials

    mu = model['mu'].reshape((-1, dyn_ndim))
    v = model['v'].reshape((-1, dyn_ndim))

    eta = mu @ a + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) x (lag, neuron) -> (time, neuron)
    r = sexp(eta + 0.5 * v @ (a ** 2))
    model['noise'] = var(y - eta, axis=0, ddof=0)  # MLE

    for obs_dim in range(obs_ndim):
        if obs_types[obs_dim] == SPIKE:
            # loading
            mu_plus_v_times_a = mu + v * a[:, obs_dim]
            grad_a = mu.T @ y[:, obs_dim] - mu_plus_v_times_a.T @ r[:, obs_dim]

            if options['hessian']:
                neghess_a = mu_plus_v_times_a.T @ (r[:, [obs_dim]] * mu_plus_v_times_a)  # + wv
                neghess_a[np.diag_indices_from(neghess_a)] += r[:, obs_dim] @ v

                try:
                    delta_a = solve(neghess_a, grad_a, sym_pos=True)
                except LinAlgError:
                    delta_a = grad_a
            else:
                delta_a = grad_a

            clip(delta_a, options['da_bound'])
            a[:, obs_dim] += options['learning_rate'] * delta_a

            # regression
            grad_b = h[obs_dim, :].T @ (y[:, obs_dim] - r[:, obs_dim])

            if options['hessian']:
                neghess_b = h[obs_dim, :].T @ (r[:, [obs_dim]] * h[obs_dim, :])
                try:
                    delta_b = solve(neghess_b, grad_b, sym_pos=True)
                except LinAlgError:
                    delta_b = grad_b
            else:
                delta_b = grad_b

            clip(delta_b, options['db_bound'])
            b[:, obs_dim] += options['learning_rate'] * delta_b
        elif obs_types[obs_dim] == LFP:
            # a's least squares solution for Gaussian channel
            # (m'm + diag(j'v))^-1 m'(y - Hb)
            tmp = mu.T @ mu
            tmp[np.diag_indices_from(tmp)] += sum(v, axis=0)
            a[:, obs_dim] = solve(tmp, mu.T @ (y[:, obs_dim] - h[obs_dim, :] @ b[:, obs_dim]), sym_pos=True)

            # b's least squares solution for Gaussian channel
            # (H'H)^-1 H'(y - ma)
            b[:, obs_dim] = solve(h[obs_dim, :].T @ h[obs_dim, :],
                                  h[obs_dim, :].T @ (y[:, obs_dim] - mu @ a[:, obs_dim]), sym_pos=True)
        else:
            pass

    # normalize loading by latent and rescale latent
    constrain_loading(model, method=options['constrain_a'], eps=options['eps'])


def hstep(model):
    """Optimize hyperparameters"""
    options = model['options']
    ntrial, nbin, dyn_ndim = model['mu'].shape
    prior = model['chol']
    rank = prior[0].shape[-1]
    mu = model['mu']
    w = model['w']
    subsample_size = options['subsample_size']
    if subsample_size is None:
        subsample_size = nbin // 2
    sigma = model['sigma']
    omega = model['omega']
    for dyn_dim in range(dyn_ndim):
        subsample = hyper.subsample(nbin, subsample_size)
        init_p = (sigma[dyn_dim] ** 2, omega[dyn_dim], 1e-3)
        bounds = ((1e-3, 1),
                  (1e-6, 1e-2),
                  (1e-4, 1e-2))
        mask = np.array([0, 1, 0])
        sigma2, omega[dyn_dim], _ = hyper.optim(options['hyper_obj'],
                                                subsample,
                                                mu[:, subsample, dyn_dim].T,
                                                w[:, subsample, dyn_dim].T,
                                                init_p,
                                                bounds,
                                                mask=mask,
                                                return_f=False)  # noise variance, small value to avoid oversmoothing
        sigma[dyn_dim] = sqrt(sigma2)
    model['chol'] = np.array(
        [ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim] for dyn_dim in range(dyn_ndim)])


def emstep(model, callback=None):
    options = model['options']
    model['it'] += 1
    with timer() as em_elapsed:
        ##########
        # E step #
        ##########
        with timer() as estep_elapsed:
            if options['learn_post']:
                for _ in range(options['e_niter']):
                    estep(model)

        ##########
        # M step #
        ##########
        with timer() as mstep_elapsed:
            if options['learn_param']:
                for _ in range(options['m_niter']):
                    mstep(model)

        ###################
        # hyperparam step #
        ###################
        with timer() as hstep_elapsed:
            if model['it'] % options['nhyper'] == 0 and options['learn_hyper']:
                hstep(model)

    model['e_elapsed'].append(estep_elapsed())
    model['m_elapsed'].append(mstep_elapsed())
    model['m_elapsed'].append(em_elapsed())

    if callable(callback):
        callback(model)


def em(model, niter, callback=None):
    options = model['options']
    eps = options['eps']
    tol = options['tol']

    ################################
    # old values
    good_mu = model['mu'].copy()
    good_w = model['w'].copy()
    good_v = model['v'].copy()
    good_a = model['a'].copy()
    good_b = model['b'].copy()
    good_noise = model['noise'].copy()
    good_sigma = model['sigma'].copy()
    good_omega = model['omega'].copy()

    stop = False

    logging_counter = 0
    last_saving_time = time.perf_counter()
    model['it']

    #######################
    # iterative algorithm #
    #######################
    gc.disable()  # disable gabbage collection during the iterative procedure
    with timer() as elapsed:
        for it in trange(niter):
            emstep(model, callback)

            # anneal learning rate
            options['learning_rate'] = 1 / (1 + options['anneal'] * model['it'])


            #####################
            # convergence check #
            #####################
            converged = norm(model['mu'].ravel() - good_mu.ravel()) <= (eps + tol * norm(good_mu.ravel())) and norm(
                model['a'].ravel() - good_a.ravel()) <= (eps + tol * norm(good_a.ravel())) and norm(
                model['b'].ravel() - good_b.ravel()) <= (eps + tol * norm(good_b.ravel()))

            copyto(good_mu, model['mu'])
            copyto(good_w, model['w'])
            copyto(good_v, model['v'])
            copyto(good_a, model['a'])
            copyto(good_b, model['b'])
            copyto(good_sigma, model['sigma'])
            copyto(good_omega, model['omega'])
            copyto(good_noise, model['noise'])

            stop = converged

            ###################################
            # statistics of current iteration #
            ###################################

            # model['ELBO'].append(lb[it])
            # model['LoadingAngle'].append(loading_angle[it])
            # model['LatentAngle'].append(latent_angle[it])
            # model['RankCorr'].append(latent_corr[it])
            # model['LL'].append(ll[it])

            # if options['verbose'] and it == 2 ** logging_counter:
            #     print('\n[{}]'.format(it))
            #     pprint(stat[it])
            #     logging_counter += 1

            now = time.perf_counter()
            if now - last_saving_time > options['saving_interval']:
                print('saving')
                save(model, 'tmp_fit.h5')
                with open('tmp_fit.pickle', 'wb') as fout:
                    pickle.dump(options, fout)
                last_saving_time = now
                print('saved')

    ##############################
    # end of iterative procedure #
    ##############################
    gc.enable()  # enable gabbage collection

    lb, ll = elbo(model)

    if options['verbose']:
        print('\nInference ends')
        print('{} iterations, ELBO: {:.4f}, elapsed: {:.3f}s\n'.format(it, lb, elapsed()))

    # model_fit['ELBO'] = lb[:it]
    # model_fit['Elapsed'] = elapsed[:it, :]
    # model_fit['LoadingAngle'] = loading_angle[:it]
    # model_fit['LatentAngle'] = latent_angle[:it]
    # model_fit['RankCorr'] = latent_corr[:it]
    # model_fit['LL'] = ll[:it]
    # model_fit['stat'] = stat
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
        eps=1e-8,
        tol=1e-5,
        evaluators=None,
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
    evaluators : list optional
        list of evaluators
    kwargs : dict, optional
        algorithm options. See fill_options()

    Returns
    -------
    dict
        fit
    """
    options = check_options(kwargs)
    options['eps'] = eps
    options['tol'] = tol
    options['niter'] += 1

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
        mu = fa.fit_transform(y.reshape((-1, obs_ndim)))
        a = fa.components_

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
    if b is None:
        b = empty((1 + lag, obs_ndim), dtype=float)
        for obs_dim in range(obs_ndim):
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
                      (1e-6, 1),
                      (1e-4, 1e-2))
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
                 options=options)

    update_w(model)
    if options['method'] == 'VB':
        update_v(model)

    inference = postprocess(em(model))
    return inference


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
    for k, v in default_options.items():
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


def constrain_loading(model, method=inf, eps=1e-8):
    mu = model['mu']
    shape_mu = mu.shape
    mu = mu.reshape((-1, mu.shape[-1]))
    a = model['a']
    if method == 'none':
        return
    if method == 'svd':
        # SVD is not good as above
        # noinspection PyTupleAssignmentBalance
        U, s, Vh = svd(a, full_matrices=False)
        model['mu'] = np.reshape(mu @ a @ Vh.T, shape_mu)
        model['a'] = Vh
    else:
        s = norm(a, ord=method, axis=1, keepdims=True) + eps
        a /= s
        mu *= s.squeeze()  # compensate latent
        model['mu'] = mu.reshape(shape_mu)


def update_w(model):
    obs_ndim, ntrial, nbin, lag1 = model['h'].shape
    dyn_ndim = model['mu'].shape[-1]

    spike_dims = model['channel'] == SPIKE
    lfp_dims = model['channel'] == LFP

    flat_mu = model['mu'].reshape((-1, dyn_ndim))
    flat_h = model['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials
    flat_v = model['v'].reshape((-1, dyn_ndim))
    shape_w = model['w'].shape

    eta = flat_mu @ model['a'] + einsum('ijk, ki -> ji', flat_h,
                                        model['b'])  # (neuron, time, lag) x (lag, neuron) -> (time, neuron)
    r = sexp(eta + 0.5 * flat_v @ (model['a'] ** 2))
    U = empty_like(r)

    U[:, spike_dims] = r[:, spike_dims]
    U[:, lfp_dims] = 1 / model['noise'][lfp_dims]
    model['w'] = reshape(U @ (model['a'].T ** 2), shape_w)


def update_v(model):
    prior = model['chol']
    rank = prior[0].shape[-1]
    eyer = identity(rank)
    ntrial, nbin, dyn_ndim = model['mu'].shape

    for trial in range(ntrial):
        w = model['w'][trial, :]
        for dyn_dim in range(dyn_ndim):
            G = prior[dyn_dim]
            GtWG = G.T @ (w[:, [dyn_dim]] * G)
            try:
                model['v'][trial, :, dyn_dim] = (
                    G * (G - G @ GtWG + G @ (GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)))).sum(axis=1)
            except LinAlgError:
                warnings.warn("singular I + G'WG")
