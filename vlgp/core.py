"""Module that does inference"""
import warnings
import gc
from pprint import pprint

import numpy as np
from sklearn.decomposition import factor_analysis
from numpy import identity, einsum, trace, inner, empty, inf, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, sum, copyto, ones
from numpy.core.umath import sqrt, PINF, log
from numpy.linalg import slogdet, LinAlgError
from scipy.linalg import lstsq, eigh, solve, norm, svd
from scipy.stats import spearmanr

from vlgp import hyper
from .evaluation import timer
from .constant import *
from .math import ichol_gauss, subspace, sexp
from .optimizer import AdamOptimizer
from .util import add_constant, rotate, lagmat

# __all__ = ['fit', 'predict']
default_options = dict(verbose=False,
                       niter=200,
                       learn_post=True,
                       learn_param=True,
                       learn_hyper=True,
                       nhyper=5,
                       e_niter=5,
                       m_niter=5,
                       decay=1,
                       hessian=True,
                       adjust_hessian=False,
                       learning_rate=0.01,
                       method='VB',
                       post_prediction=True,
                       update_bound=1.0,
                       backtrack=False,
                       subsample_size=None,
                       hyper_obj='ELBO',
                       Adam=False,
                       gp_noise=1e-3,
                       constrain_mu=True,
                       constrain_a=inf,
                       dmu_bound=1.0,
                       da_bound=1.0,
                       db_bound=5.0)


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

    eye_rank = identity(rank)

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

    llspike = sum(y[:, spike_dims] * eta[:, spike_dims] - r[:, spike_dims])  # verified by predict()

    # noinspection PyTypeChecker
    lllfp = - 0.5 * sum(
        ((y[:, lfp_dims] - eta[:, lfp_dims]) ** 2 + v @ (a[:, lfp_dims] ** 2)) / noise[lfp_dims] + log(noise[lfp_dims]))

    ll = llspike + lllfp

    lb = ll

    for trial in range(ntrial):
        mu = model_fit['mu'][trial, :]
        w = model_fit['w'][trial, :]
        for dyn_dim in range(dyn_ndim):
            G = prior[dyn_dim]
            GtWG = G.T @ (w[:, [dyn_dim]] * G)
            tmp = GtWG @ solve(eye_rank + GtWG, GtWG, sym_pos=True)  # expected to be nonsingular
            # TODO: Need a better approximate of mu^T K^{-1} mu than least squares.
            G_mldiv_mu = lstsq(G, mu[:, dyn_dim])[0]
            mu_Kinv_mu = inner(G_mldiv_mu, G_mldiv_mu)
            # K = G @ G.T
            # mu_Kinv_mu = mu[:, l] @ spilu(csc_matrix(K)).solve(mu[:, l])
            tr = nbin - trace(GtWG) + trace(tmp)
            lndet = slogdet(eye_rank - GtWG + tmp)[1]

            lb += -0.5 * mu_Kinv_mu - 0.5 * tr + 0.5 * lndet + 0.5 * nbin

    return lb, ll


def accumulate(acc, grad, b=1):
    """
    Accumulate second moment of gradient for adjusting Hessian

    Parameters
    ----------
    acc : ndarray
        accumulation
    grad : ndarray
        gradient
    b : double
        decay rate

    Returns
    -------
    ndarray
        sum of second moments
    """
    if b < 1:
        return b * acc + (1 - b) * grad ** 2
    else:
        return acc + grad ** 2


def infer(model_fit, options):
    """
    Inference procedure

    Parameters
    ----------
    model_fit : dict
        initial model fit
    options : dict

    Returns
    -------
    dict
        model fit
    """

    def estep():
        """Optimize posterior (E step)"""
        obs_ndim = model_fit['y'].shape[-1]
        ntrial, nbin, dyn_ndim = model_fit['mu'].shape
        prior = model_fit['chol']
        rank = prior[0].shape[-1]
        a = model_fit['a']
        b = model_fit['b']
        noise = model_fit['noise']

        eyer = identity(rank)
        residual = empty((nbin, obs_ndim), dtype=float)
        U = empty((nbin, obs_ndim), dtype=float)

        for trial in range(ntrial):
            # trial slices
            y = model_fit['y'][trial, :]
            h = model_fit['h'][:, trial, :, :]
            mu = model_fit['mu'][trial, :]
            w = model_fit['w'][trial, :]
            v = model_fit['v'][trial, :]

            hb = einsum('ijk, ki -> ji', h, b)
            eta = mu @ a + hb
            r = sexp(eta + 0.5 * v @ (a ** 2))
            for dyn_dim in range(dyn_ndim):
                G = prior[dyn_dim]

                residual[:, spike_dims] = y[:, spike_dims] - r[:, spike_dims]  # residuals of Poisson observations
                residual[:, lfp_dims] = (y[:, lfp_dims] - eta[:, lfp_dims]) / noise[
                    lfp_dims]  # residuals of Gaussian observations

                if adjust_hessian:
                    grad_mu_resid = (y[:, spike_dims] - r[:, spike_dims]) @ a[dyn_dim, spike_dims] + \
                                    ((y[:, lfp_dims] - eta[:, lfp_dims]) / noise[lfp_dims]) @ a[dyn_dim, lfp_dims]
                    grad_mu = grad_mu_resid - lstsq(G.T, lstsq(G, mu[:, dyn_dim])[0])[0]
                    dmu_acc[trial, :, dyn_dim] = accumulate(dmu_acc[trial, :, dyn_dim], grad_mu, decay)
                    wadj = w[:, dyn_dim] + eps + sqrt(dmu_acc[trial, :, dyn_dim])  # adjusted Hessian
                else:
                    wadj = w[:, dyn_dim]  # keep dimension
                wadj = wadj[:, None]
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[dyn_dim, :])) - mu[:, dyn_dim]
                delta_mu = u - G @ ((wadj * G).T @ u) + \
                           G @ (GtWG @ solve(eyer + GtWG, (wadj * G).T @ u, sym_pos=True))

                clip(delta_mu, options['dmu_bound'])
                if options['Adam']:
                    optimizer = options['optimizer_mu'][trial, dyn_dim]
                    delta_mu = optimizer.next_update(delta_mu)
                mu[:, dyn_dim] += delta_mu

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
                        v[:, dyn_dim] = (G * (G - G @ GtWG + G @ (GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)))).sum(
                            axis=1)
                    except LinAlgError:
                        warnings.warn("singular I + G'WG")

        # center over all trials if not only infer posterior
        if options['constrain_mu']:
            shape = model_fit['mu'].shape
            mu_over_trials = model_fit['mu'].reshape((-1, dyn_ndim))
            mean_over_trials = mu_over_trials.mean(axis=0)
            model_fit['b'][0, :] += mean_over_trials @ model_fit['a']  # compensate bias
            mu_over_trials -= mean_over_trials
            model_fit['mu'] = mu_over_trials.reshape(shape)

    def mstep():
        """Optimize loading and regression (M step)"""
        obs_ndim, ntrial, nbin, lag1 = model_fit['h'].shape  # neuron, trial, time, lag
        ntrial, nbin, dyn_ndim = model_fit['mu'].shape
        obs_types = model_fit['channel']
        a = model_fit['a']
        b = model_fit['b']

        y = model_fit['y'].reshape((-1, obs_ndim))  # concatenate trials
        h = model_fit['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials

        mu = model_fit['mu'].reshape((-1, dyn_ndim))
        v = model_fit['v'].reshape((-1, dyn_ndim))

        eta = mu @ a + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) x (lag, neuron) -> (time, neuron)
        r = sexp(eta + 0.5 * v @ (a ** 2))
        model_fit['noise'] = var(y - eta, axis=0, ddof=0)  # MLE

        for obs_dim in range(obs_ndim):
            if obs_types[obs_dim] == SPIKE:
                # loading
                mu_plus_v_times_a = mu + v * a[:, obs_dim]
                grad_a = mu.T @ y[:, obs_dim] - mu_plus_v_times_a.T @ r[:, obs_dim]

                if options['hessian']:
                    neghess_a = mu_plus_v_times_a.T @ (r[:, [obs_dim]] * mu_plus_v_times_a)  # + wv
                    neghess_a[np.diag_indices_from(neghess_a)] += r[:, obs_dim] @ v

                    if adjust_hessian:
                        da_acc[:, obs_dim] = accumulate(da_acc[:, obs_dim], grad_a, decay)
                        neghess_a[np.diag_indices_from(neghess_a)] += eps + sqrt(da_acc[:, obs_dim])
                    try:
                        delta_a = solve(neghess_a, grad_a, sym_pos=True)
                    except LinAlgError:
                        # print('singular Hessian a')
                        delta_a = grad_a
                else:
                    delta_a = grad_a

                clip(delta_a, options['da_bound'])
                if options['Adam']:
                    optimizer_a = options['optimizer_a'][obs_dim]
                    delta_a = optimizer_a.next_update(delta_a)
                a[:, obs_dim] += delta_a

                # regression
                grad_b = h[obs_dim, :].T @ (y[:, obs_dim] - r[:, obs_dim])

                if options['hessian']:
                    neghess_b = h[obs_dim, :].T @ (r[:, [obs_dim]] * h[obs_dim, :])
                    # TODO: inactive neurons never fire across all trials which may cause zero Hessian
                    if adjust_hessian:
                        db_acc[:, obs_dim] = accumulate(db_acc[:, obs_dim], grad_b, decay)
                        neghess_b[np.diag_indices_from(neghess_b)] += eps + sqrt(db_acc[:, obs_dim])
                    try:
                        delta_b = solve(neghess_b, grad_b, sym_pos=True)
                    except LinAlgError:
                        # print('singular Hessian b')
                        delta_b = grad_b
                else:
                    delta_b = grad_b
                clip(delta_b, options['db_bound'])
                if options['Adam']:
                    optimizer_b = options['optimizer_b'][obs_dim]
                    delta_b = optimizer_b.next_update(delta_b)
                b[:, obs_dim] += delta_b
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
        constrain_loading(model_fit, method=options['constrain_a'], eps=options['eps'])

    def hstep():
        """Optimize hyperparameters"""
        ntrial, nbin, dyn_ndim = model_fit['mu'].shape
        prior = model_fit['chol']
        rank = prior[0].shape[-1]
        mu = model_fit['mu']
        w = model_fit['w']
        subsample_size = options['subsample_size']
        if subsample_size is None:
            subsample_size = nbin // 2
        sigma = model_fit['sigma']
        omega = model_fit['omega']
        multiplier = 10.0
        for dyn_dim in range(dyn_ndim):
            # subsample = np.random.choice(nbin, subsample_size, replace=False)
            subsample = hyper.subsample(nbin, subsample_size)
            # subsample = np.arange(subsample_size) + np.random.randint(nbin - subsample_size)
            init_p = (sigma[dyn_dim] ** 2, omega[dyn_dim], 1e-3)
            bounds = ((1e-3, 1),
                      (max(1e-6, omega[dyn_dim] / multiplier), min(1.0, omega[dyn_dim] * multiplier)),
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
        copyto(good_sigma, sigma)
        copyto(good_omega, omega)
        model_fit['chol'] = np.array(
            [ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim] for dyn_dim in range(dyn_ndim)])

    #################
    # function body #
    #################
    # truth
    x = model_fit.get('x')
    alpha = model_fit.get('alpha')

    # options
    eps = options['eps']
    tol = options['tol']
    decay = options['decay']
    adjust_hessian = options['adjust_hessian']
    dmu_acc = options['dmu_acc']
    da_acc = options['da_acc']
    db_acc = options['db_acc']

    spike_dims = model_fit['channel'] == SPIKE
    lfp_dims = model_fit['channel'] == LFP

    ################################
    # old values
    good_mu = model_fit['mu'].copy()
    good_w = model_fit['w'].copy()
    good_v = model_fit['v'].copy()
    good_a = model_fit['a'].copy()
    good_b = model_fit['b'].copy()
    good_noise = model_fit['noise'].copy()
    good_sigma = model_fit['sigma'].copy()
    good_omega = model_fit['omega'].copy()

    stat = empty(options['niter'], dtype=object)
    lb = zeros(options['niter'], dtype=float)
    ll = zeros(options['niter'], dtype=float)
    elapsed = zeros((options['niter'], 3), dtype=float)
    loading_angle = zeros(options['niter'], dtype=float)
    latent_angle = zeros(options['niter'], dtype=float)
    latent_corr = zeros((options['niter'], model_fit['mu'].shape[-1]), dtype=float)

    # iteration 0
    # lb[0], ll[0] = elbo(obj)
    lb[0], ll[0] = np.finfo(float).min, np.finfo(float).min
    if alpha is not None:
        loading_angle[0] = subspace(alpha.T, model_fit['a'].T)
    if x is not None:
        rotated = empty_like(x, dtype=float)
        # rotate trial by trial
        for itrial in range(x.shape[0]):
            rotated[itrial, :] = rotate(add_constant(model_fit['mu'][itrial, :]), x[itrial, :])
        latent_angle[0] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
        rho, _ = spearmanr(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
        latent_corr[0] = rho[np.arange(x.shape[-1]), np.arange(x.shape[-1]) + x.shape[-1]]

    # iterative algorithm
    it = 1  # iteration counter
    stop = False

    logging_counter = 0

    if options['verbose']:
        print('\nstarting')

    #######################
    # iterative algorithm #
    #######################
    gc.disable()  # disable gabbage collection during the iterative procedure
    with timer() as algo_elapsed:
        while not stop and it < options['niter']:
            with timer() as iter_elapsed:
                ##########
                # E step #
                ##########
                with timer() as estep_elapsed:
                    if options['learn_post']:
                        for _ in range(options['e_niter']):
                            estep()

                ##########
                # M step #
                ##########
                with timer() as mstep_elapsed:
                    if options['learn_param']:
                        for _ in range(options['m_niter']):
                            mstep()

                ###################
                # hyperparam step #
                ###################
                with timer() as hstep_elapsed:
                    if it % options['nhyper'] == 0 and options['learn_hyper']:
                        hstep()

            # Calculate angle between latent subspace if true latent is given.
            if x is not None:
                for itrial in range(x.shape[0]):
                    rotated[itrial, :] = rotate(add_constant(model_fit['mu'][itrial, :]), x[itrial, :])
                latent_angle[it] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
                rho, _ = spearmanr(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
                latent_corr[it] = rho[np.arange(x.shape[-1]), np.arange(x.shape[-1]) + x.shape[-1]]

            # Calculate angle between loading subspace if true loading is given.
            if alpha is not None:
                loading_angle[it] = subspace(alpha.T, model_fit['a'].T)

            #####################
            # convergence check #
            #####################
            lb[it], ll[it] = 0, 0
            converged = norm(model_fit['mu'].ravel() - good_mu.ravel()) <= (eps + tol * norm(good_mu.ravel())) and norm(
                model_fit['a'].ravel() - good_a.ravel()) <= (eps + tol * norm(good_a.ravel())) and norm(
                model_fit['b'].ravel() - good_b.ravel()) <= (eps + tol * norm(good_b.ravel()))

            copyto(good_mu, model_fit['mu'])
            copyto(good_w, model_fit['w'])
            copyto(good_v, model_fit['v'])
            copyto(good_a, model_fit['a'])
            copyto(good_b, model_fit['b'])
            copyto(good_noise, model_fit['noise'])

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
            stat[it]['sigma'] = good_sigma
            stat[it]['omega'] = good_omega

            # TODO: change stat to OrderedDict
            if options['verbose'] and it == 2 ** logging_counter:
                print('\n[{}]'.format(it))
                pprint(stat[it])
                # for k in sorted(stat[it]):
                #     pp.print('{}: {}'.format(k, stat[it][k]))
                logging_counter += 1

            it += 1
    ##############################
    # end of iterative procedure #
    ##############################
    gc.enable()  # enable gabbage collection

    lb[it - 1], ll[it - 1] = elbo(model_fit)
    if options['verbose']:
        print('\nInference ends')
        print('{} iterations, ELBO: {:.4f}, elapsed: {:.3f}s\n'.format(it - 1, lb[it - 1], algo_elapsed()))

    model_fit['ELBO'] = lb[:it]
    model_fit['Elapsed'] = elapsed[:it, :]
    model_fit['LoadingAngle'] = loading_angle[:it]
    model_fit['LatentAngle'] = latent_angle[:it]
    model_fit['RankCorr'] = latent_corr[:it]
    model_fit['LL'] = ll[:it]
    model_fit['stat'] = stat
    return model_fit


def check_obs_type(types):
    types = asarray(types)
    coded_types = np.empty_like(types, dtype=int)
    for i, type in enumerate(types):
        if type == 'spike':
            coded_types[i] = SPIKE
        elif type == 'lfp':
            coded_types[i] = LFP
        else:
            coded_types[i] = UNUSED
    return coded_types


def fit(y,
        obs_types,
        dyn_ndim,
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
        evaluators=[],
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
    gp_noise : double optional
        noise variance in GP kernel
    evaluators : list optional
        list of evaluators
    kwargs : dict, optional
        algorithm options. See fill_options()

    Returns
    -------
    dict
        fit
    """
    options = check_options(**kwargs)
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

    # prior = empty(dyn_ndim, dtype=np.ndarray)
    prior = np.array([ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim] for dyn_dim in range(dyn_ndim)])
    # chol = empty((dyn_ndim, nbin, rank), dtype=float)
    # for dyn_dim in range(dyn_ndim):
    #     chol[dyn_dim, :] = ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim]

    ##############

    # initialize bias and autoregression
    if b is None:
        b = empty((1 + lag, obs_ndim), dtype=float)
        for obs_dim in range(obs_ndim):
            b[:, obs_dim] = \
                lstsq(h.reshape((obs_ndim, -1, 1 + lag))[obs_dim, :], y.reshape((-1, obs_ndim))[:, obs_dim])[0]

    # initialize noises of guassian channels
    noise = var(y.reshape((-1, obs_ndim)), axis=0, ddof=0)

    # w and v
    w = 0 * ones((ntrial, nbin, dyn_ndim), dtype=float)
    if options['method'] == 'VB':
        v = np.repeat(sigma[newaxis, ...], ntrial * nbin, axis=1).reshape((ntrial, nbin, dyn_ndim))
    else:
        v = 0 * ones((ntrial, nbin, dyn_ndim), dtype=float)

    model_fit = dict(y=y,
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
                     beta=beta)

    options['dmu_acc'] = zeros_like(mu)
    options['da_acc'] = zeros_like(a)
    options['db_acc'] = zeros_like(b)

    if options['Adam']:
        options['optimizer_mu'] = empty((ntrial, dyn_ndim), dtype=object)
        options['optimizer_a'] = empty(obs_ndim, dtype=object)
        options['optimizer_b'] = empty(obs_ndim, dtype=object)

        for each in np.nditer(options['optimizer_mu'], flags=['refs_ok'], op_flags=['readwrite']):
            each[...] = AdamOptimizer(nbin, options['learning_rate'])
        for each in np.nditer(options['optimizer_a'], flags=['refs_ok'], op_flags=['readwrite']):
            each[...] = AdamOptimizer(dyn_ndim, options['learning_rate'])
        for each in np.nditer(options['optimizer_b'], flags=['refs_ok'], op_flags=['readwrite']):
            each[...] = AdamOptimizer(b.shape[0], options['learning_rate'])

    inference = postprocess(infer(model_fit, options))
    return inference, options


def check_options(**kwargs):
    """
    Fill missing options with default values

    Parameters
    ----------
    options : dict
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


def postprocess(model_fit):
    """
    Remove intermediate and empty variables, and compute decomposition of posterior covariance.

    Parameters
    ----------
    model_fit : dict
        raw fit

    Returns
    -------
    dict
        fit that contains prior, posterior, loading and regression
    """
    ntrial, nbin, dyn_ndim = model_fit['mu'].shape
    prior = model_fit['chol']
    rank = prior[0].shape[-1]
    w = model_fit['w']
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
    model_fit['L'] = L
    keys = list(model_fit.keys())
    for key in keys:
        if model_fit.get(key, None) is None:
            model_fit.pop(key, None)
    model_fit.pop('h', None)
    model_fit.pop('stat', None)
    return model_fit


def predict(y, x, a, b, v=None):
    """
    Predict firing rate

    Parameters
    ----------
    y : ndarray
        spike trains
    x : ndarray
        posterior mean
    a : ndarray
        loading
    b : ndarray
        regression
    v : ndarray
        posterior variance

    Returns
    -------
    ndarray
        predicted firing rate
    """
    ntrial, nbin, obs_ndim = y.shape
    dyn_ndim = x.shape[-1]
    lag = b.shape[0] - 1

    # regression (h dot b) part
    reg = empty_like(y)
    for obs_dim in range(obs_ndim):
        for trial in range(ntrial):
            h = add_constant(lagmat(y[trial, :, obs_dim], lag=lag))
            reg[trial, :, obs_dim] = h @ b[:, obs_dim]
    eta = x.reshape((-1, dyn_ndim)) @ a + reg.reshape((-1, obs_ndim))
    r = np.exp(eta + 0.5 * v.reshape((-1, dyn_ndim)) @ (a ** 2)) if v is not None else np.exp(eta)
    yhat = r.reshape(y.shape)
    return yhat


def clip(delta, lbound, ubound=None):
    if ubound is None:
        assert (lbound > 0)
        ubound = lbound
        lbound = -lbound
    else:
        assert ubound > lbound
    np.clip(delta, lbound, ubound, out=delta)


def constrain_loading(model_fit, method=inf, eps=1e-8):
    mu = model_fit['mu']
    shape_mu = mu.shape
    mu = mu.reshape((-1, mu.shape[-1]))
    a = model_fit['a']
    if method == 'none':
        return
    if method == 'svd':
        # SVD is not good as above
        # noinspection PyTupleAssignmentBalance
        U, s, Vh = svd(a, full_matrices=False)
        model_fit['mu'] = np.reshape(mu @ a @ Vh.T, shape_mu)
        model_fit['a'] = Vh
    else:
        s = norm(a, ord=method, axis=1, keepdims=True) + eps
        a /= s
        mu *= s.squeeze()  # compensate latent
        model_fit['mu'] = mu.reshape(shape_mu)
