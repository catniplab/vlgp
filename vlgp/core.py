"""Module that does inference"""
# TODO: Find a way to replace diagonal matrix construction (np.diag) in loop.
import timeit
import warnings

import numpy as np
from sklearn.decomposition import factor_analysis
from numpy import identity, einsum, trace, inner, empty, inf, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, sum, copyto, ones
from numpy.core.umath import sqrt, PINF, log
from numpy.linalg import slogdet, LinAlgError
from scipy.linalg import lstsq, eigh, solve, norm
from scipy.stats import spearmanr

from vlgp import hyper
from .math import ichol_gauss, subspace, sexp
from .optimizer import AdamOptimizer
from .util import add_constant, rotate, lagmat


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
    dyn_ndim, _, rank = model_fit['chol'].shape  # latent, time, rank

    eye_rank = identity(rank)

    y = model_fit['y'].reshape((-1, obs_ndim))  # concatenate trials
    h = model_fit['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials
    obs_types = model_fit['channel']

    chol = model_fit['chol']

    mu = model_fit['mu'].reshape((-1, dyn_ndim))
    v = model_fit['v'].reshape((-1, dyn_ndim))

    a = model_fit['a']
    b = model_fit['b']
    noise = model_fit['noise']

    spike = obs_types == 'spike'
    lfp = obs_types == 'lfp'

    eta = mu @ a + einsum('ijk, ki -> ji', h.reshape((obs_ndim, nbin * ntrial, lag1)), b)
    frate = sexp(eta + 0.5 * v @ (a ** 2))

    llspike = sum(y[:, spike] * eta[:, spike] - frate[:, spike])  # verified by predict()

    # noinspection PyTypeChecker
    lllfp = - 0.5 * sum(((y[:, lfp] - eta[:, lfp]) ** 2 + v @ (a[:, lfp] ** 2)) / noise[lfp] + log(noise[lfp]))

    ll = llspike + lllfp

    lb = ll

    for trial in range(ntrial):
        mu = model_fit['mu'][trial, :]
        w = model_fit['w'][trial, :]
        for dyn_dim in range(dyn_ndim):
            G = chol[dyn_dim, :]
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
        obs_ndim, ntrial, nbin, lag1 = model_fit['h'].shape  # neuron, trial, time, lag
        dyn_ndim, nbin, rank = model_fit['chol'].shape
        obs_types = model_fit['channel']
        chol = model_fit['chol']
        a = model_fit['a']
        b = model_fit['b']
        noise = model_fit['noise']

        spike = obs_types == 'spike'
        lfp = obs_types == 'lfp'

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
            rate = sexp(eta + 0.5 * v @ (a ** 2))
            for dyn_dim in range(dyn_ndim):
                # lam, eta = firing_rate(mu, v, a, b, h)
                G = chol[dyn_dim, :, :]

                residual[:, spike] = y[:, spike] - rate[:, spike]  # residuals of Poisson observations
                residual[:, lfp] = (y[:, lfp] - eta[:, lfp]) / noise[lfp]  # residuals of Gaussian observations

                if adjust_hessian:
                    grad_mu_resid = (y[:, spike] - rate[:, spike]) @ a[dyn_dim, spike] + \
                                    ((y[:, lfp] - eta[:, lfp]) / noise[lfp]) @ a[dyn_dim, lfp]
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

                check_update(delta_mu)
                if options['Adam']:
                    optimizer = options['optimizer_mu'][trial, dyn_dim]
                    delta_mu = optimizer.update(delta_mu)
                mu[:, dyn_dim] += delta_mu

            eta = mu @ a + hb
            rate = sexp(eta + 0.5 * v @ (a ** 2))
            U[:, spike] = rate[:, spike]
            U[:, lfp] = 1 / noise[lfp]
            copyto(w, U @ (a.T ** 2))
            if options['method'] == 'VB':
                for dyn_dim in range(dyn_ndim):
                    G = chol[dyn_dim, :, :]
                    GtWG = G.T @ (w[:, [dyn_dim]] * G)
                    try:
                        v[:, dyn_dim] = (G * (G - G @ GtWG + G @ (GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)))).sum(
                            axis=1)
                    except LinAlgError:
                        warnings.warn("singular I + G'WG")

        # center over all trials if not only infer posterior
        if options['learn_param']:
            shape = model_fit['mu'].shape
            mu_over_trials = model_fit['mu'].reshape((-1, dyn_ndim))
            mean_over_trials = mu_over_trials.mean(axis=0)
            model_fit['b'][0, :] += mean_over_trials @ model_fit['a']  # compensate bias
            mu_over_trials -= mean_over_trials
            model_fit['mu'] = mu_over_trials.reshape(shape)

    def mstep():
        """Optimize loading and regression (M step)"""
        # TODO: Update eta and frate neuron-wise
        obs_ndim, ntrial, nbin, lag1 = model_fit['h'].shape  # neuron, trial, time, lag
        dyn_ndim, nbin, rank = model_fit['chol'].shape
        obs_types = model_fit['channel']
        a = model_fit['a']
        b = model_fit['b']

        y = model_fit['y'].reshape((-1, obs_ndim))  # concatenate trials
        h = model_fit['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials

        mu = model_fit['mu'].reshape((-1, dyn_ndim))
        v = model_fit['v'].reshape((-1, dyn_ndim))

        eta = mu @ a + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) x (lag, neuron) -> (time, neuron)
        rate = sexp(eta + 0.5 * v @ (a ** 2))
        model_fit['noise'] = var(y - eta, axis=0, ddof=0)  # MLE

        for obs_dim in range(obs_ndim):
            # eta = mu @ a + einsum('ijk, ki -> ji', h, b)
            # lam = sexp(eta + 0.5 * v @ (a ** 2))
            if obs_types[obs_dim] == 'spike':
                # loading
                # va = v * a[:, obs_dim]  # (nbin, dyn_ndim)
                mu_plus_v_times_a = mu + v * a[:, obs_dim]
                # wv = diag(frate[:, obs_dim] @ v)
                # wv = rate[:, obs_dim] @ v

                grad_a = mu.T @ y[:, obs_dim] - mu_plus_v_times_a.T @ rate[:, obs_dim]

                if options['hessian']:
                    neghess_a = mu_plus_v_times_a.T @ (rate[:, [obs_dim]] * mu_plus_v_times_a)  # + wv
                    neghess_a[np.diag_indices_from(neghess_a)] += rate[:, obs_dim] @ v

                    if adjust_hessian:
                        da_acc[:, obs_dim] = accumulate(da_acc[:, obs_dim], grad_a, decay)
                        neghess_a[np.diag_indices_from(neghess_a)] += eps + sqrt(da_acc[:, obs_dim])
                    try:
                        delta_a = solve(neghess_a, grad_a, sym_pos=True)
                    except LinAlgError:
                        print('singular Hessian a')
                        delta_a = grad_a
                else:
                    delta_a = grad_a

                check_update(delta_a)
                if options['Adam']:
                    optimizer_a = options['optimizer_a'][obs_dim]
                    delta_a = optimizer_a.update(delta_a)
                a[:, obs_dim] += delta_a

                # regression
                grad_b = h[obs_dim, :].T @ (y[:, obs_dim] - rate[:, obs_dim])

                if options['hessian']:
                    neghess_b = h[obs_dim, :].T @ (rate[:, [obs_dim]] * h[obs_dim, :])
                    # TODO: inactive neurons never fire across all trials which may cause zero Hessian
                    if adjust_hessian:
                        db_acc[:, obs_dim] = accumulate(db_acc[:, obs_dim], grad_b, decay)
                        neghess_b[np.diag_indices_from(neghess_b)] += eps + sqrt(db_acc[:, obs_dim])
                    try:
                        delta_b = solve(neghess_b, grad_b, sym_pos=True)
                    except LinAlgError:
                        print('singular Hessian b')
                        delta_b = grad_b
                else:
                    delta_b = grad_b
                # check_update(delta_b)
                if options['Adam']:
                    optimizer_b = options['optimizer_b'][obs_dim]
                    delta_b = optimizer_b.update(delta_b)
                b[:, obs_dim] += delta_b
            elif obs_types[obs_dim] == 'lfp':
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
                raise ValueError('Unsupported channel')

        # normalize loading by latent and rescale latent
        if options['learn_post']:
            scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
            a /= scale
            mu *= scale.squeeze()  # compensate latent
            model_fit['mu'] = mu.reshape(model_fit['mu'].shape)
            # SVD is not good as above
            # noinspection PyTupleAssignmentBalance
            # U, s, Vh = svd(a, full_matrices=False)
            # obj['mu'] = np.reshape(mu @ a @ Vh.T, (ntrial, nbin, dyn_ndim))
            # a[:] = Vh

    def check_update(delta):
        np.clip(delta, -options['update_bound'], options['update_bound'], out=delta)
        if np.any(delta < -options['update_bound']) or np.any(delta > options['update_bound']):
            warnings('update overflow')

    def hstep():
        """Optimize hyperparameters"""
        dyn_ndim, nbin, rank = model_fit['chol'].shape
        mu = model_fit['mu']
        w = model_fit['w']
        subsample_size = options['subsample_size']
        if subsample_size is None:
            subsample_size = nbin // 2
        omega = model_fit['omega']
        multiplier = 10.0
        for dyn_dim in range(dyn_ndim):
            # subsample = np.random.choice(nbin, subsample_size, replace=False)
            subsample = hyper.subsample(nbin, subsample_size)
            # subsample = np.arange(subsample_size) + np.random.randint(nbin - subsample_size)
            init_p = (omega[dyn_dim], 1e-3)
            bounds = ((max(1e-6, omega[dyn_dim] / multiplier), min(1.0, omega[dyn_dim] * multiplier)),
                      (1e-4, 1e-2))
            mask = np.array([1, 0])
            # bounds = ((1e-6, 10), (1e-6, 1))
            omega[dyn_dim], _ = hyper.optim(options['hyper_obj'],
                                            subsample,
                                            mu[:, subsample, dyn_dim].T,
                                            w[:, subsample, dyn_dim].T,
                                            init_p,
                                            bounds,
                                            mask=mask,
                                            return_f=False)  # noise variance, small value to avoid oversmoothing

        # copyto(good_sigma, sigma)
        # print('update omega: {}'.format(model_fit['omega']))
        copyto(good_omega, omega)
        for dyn_dim in range(dyn_ndim):
            model_fit['chol'][dyn_dim, :] = ichol_gauss(nbin, model_fit['omega'][dyn_dim], rank) * \
                                            model_fit['sigma'][dyn_dim]

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
    infer_tick = timeit.default_timer()

    logging_counter = 0

    if options['verbose']:
        print('\nstarting')

    while not stop and it < options['niter']:
        iter_tick = timeit.default_timer()

        ##########
        # E step #
        ##########
        e_tick = timeit.default_timer()
        if options['learn_post']:
            for _ in range(options['e_niter']):
                estep()
        # elbo(obj)
        e_tock = timeit.default_timer()
        elapsed[it, 0] = e_tock - e_tick

        # Calculate angle between latent subspace if true latent is given.
        if x is not None:
            for itrial in range(x.shape[0]):
                rotated[itrial, :] = rotate(add_constant(model_fit['mu'][itrial, :]), x[itrial, :])
            latent_angle[it] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
            rho, _ = spearmanr(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
            latent_corr[it] = rho[np.arange(x.shape[-1]), np.arange(x.shape[-1]) + x.shape[-1]]

        ##########
        # M step #
        ##########
        m_tick = timeit.default_timer()
        if options['learn_param']:
            for _ in range(options['m_niter']):
                mstep()
        m_tock = timeit.default_timer()
        elapsed[it, 1] = m_tock - m_tick

        # Calculate angle between loading subspace if true loading is given.
        if alpha is not None:
            loading_angle[it] = subspace(alpha.T, model_fit['a'].T)

        #####################
        # convergence check #
        #####################
        lb[it], ll[it] = 0, 0
        # decreased = lb[it] < lb[it - 1]
        # converged = np.allclose(model_fit['mu'], good_mu)
        converged = norm(model_fit['mu'].ravel() - good_mu.ravel()) <= (eps + tol * norm(good_mu.ravel())) and norm(
            model_fit['a'].ravel() - good_a.ravel()) <= (eps + tol * norm(good_a.ravel())) and norm(
            model_fit['b'].ravel() - good_b.ravel()) <= (eps + tol * norm(good_b.ravel()))

        # if decreased:
        #     # if options['verbose']:
        #     #     print('\nELBO decreased.')
        #     if options['backtrack']:
        #         copyto(model_fit['mu'], good_mu)
        #         copyto(model_fit['w'], good_w)
        #         copyto(model_fit['v'], good_v)
        #         copyto(model_fit['a'], good_a)
        #         copyto(model_fit['b'], good_b)
        #         copyto(model_fit['noise'], good_noise)
        #         lb[it] = lb[it - 1]
        #     else:
        #         copyto(good_mu, model_fit['mu'])
        #         copyto(good_w, model_fit['w'])
        #         copyto(good_v, model_fit['v'])
        #         copyto(good_a, model_fit['a'])
        #         copyto(good_b, model_fit['b'])
        #         copyto(good_noise, model_fit['noise'])
        # else:
        copyto(good_mu, model_fit['mu'])
        copyto(good_w, model_fit['w'])
        copyto(good_v, model_fit['v'])
        copyto(good_a, model_fit['a'])
        copyto(good_b, model_fit['b'])
        copyto(good_noise, model_fit['noise'])

        # converged = converged  # or abs(lb[iiter] - lb[iiter - 1]) < options['tol'] * abs(lb[iiter - 1])
        # stop = converged or decreased
        stop = converged

        ###################
        # hyperparam step #
        ###################
        if it % options['nhyper'] == 0 and options['learn_hyper']:
            # lb[it], ll[it] = elbo(model_fit)
            hstep()

        ###################################
        # statistics of current iteration #
        ###################################
        iter_tock = timeit.default_timer()
        elapsed[it, 2] = iter_tock - iter_tick

        stat[it] = {}
        stat[it]['Elapsed Post'] = elapsed[it, 0]
        stat[it]['Elapsed Param'] = elapsed[it, 1]
        stat[it]['Elapsed Total'] = elapsed[it, 2]
        # stat[it]['ELBO'] = lb[it]
        # stat[it]['LL'] = ll[it]
        # stat[it]['sigma'] = good_sigma
        stat[it]['omega'] = good_omega

        # TODO: change stat to OrderedDict
        if options['verbose'] and it == 2 ** logging_counter:
            print('\n[{}]'.format(it))
            for k in sorted(stat[it]):
                print('{}: {}'.format(k, stat[it][k]))
            logging_counter += 1

        it += 1
    infer_tock = timeit.default_timer()

    lb[it - 1], ll[it - 1] = elbo(model_fit)
    if options['verbose']:
        print('\nInference ends')
        print('{} iterations, ELBO: {:.4f}, elapsed: {:.3f}\n'.format(it - 1,
                                                                      lb[it - 1],
                                                                      infer_tock - infer_tick))
    model_fit['ELBO'] = lb[:it]
    model_fit['Elapsed'] = elapsed[:it, :]
    model_fit['LoadingAngle'] = loading_angle[:it]
    model_fit['LatentAngle'] = latent_angle[:it]
    model_fit['RankCorr'] = latent_corr[:it]
    model_fit['LL'] = ll[:it]
    model_fit['stat'] = stat
    return model_fit


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
    kwargs : dict, optional
        algorithm options. See fill_options()

    Returns
    -------
    dict
        fit
    """
    options = fill_options(kwargs)
    options['eps'] = eps
    options['tol'] = tol

    y = asarray(y)
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]

    obs_types = asarray(obs_types)
    ntrial, nbin, obs_ndim = y.shape

    # make design matrix of regression
    h = empty((obs_ndim, ntrial, nbin, 1 + lag), dtype=float)
    for ichannel in range(obs_ndim):
        for itrial in range(ntrial):
            h[ichannel, itrial, :] = add_constant(lagmat(y[itrial, :, ichannel], lag=lag))

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent
    if a is None and mu is None:
        fa = factor_analysis.FactorAnalysis(n_components=dyn_ndim, svd_method='lapack')
        mu = fa.fit_transform(y.reshape((-1, obs_ndim)))
        a = fa.components_

        # constrain loading and center latent
        mu -= mu.mean(axis=0)
        scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
        a /= scale
        mu *= scale.squeeze()  # compensate latent
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
            omega_opt = np.zeros_like(omega_grid)
            fval_opt = np.zeros_like(omega_grid)

            bounds = ((1e-6, 1),
                      (1e-4, 1e-2))
            mask = np.array([1, 0])
            for i, o in enumerate(omega_grid):
                init_p = (o, options['gp_noise'])
                (omega_opt[i], _), fval_opt[i] = hyper.optim('GP',
                                                             subsample,  # time
                                                             mu[:, subsample, dyn_dim].T,
                                                             None,  # Sigma
                                                             init_p,
                                                             bounds=bounds,
                                                             mask=mask,
                                                             return_f=True)
            idx = np.argmin(fval_opt)
            omega[dyn_dim] = omega_opt[idx]

    # make Cholesky of prior
    if rank is None:
        rank = nbin
    chol = empty((dyn_ndim, nbin, rank), dtype=float)
    for dyn_dim in range(dyn_ndim):
        chol[dyn_dim, :] = ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim]

    ##############

    # initialize bias and autoregression
    if b is None:
        b = empty((1 + lag, obs_ndim), dtype=float)
        for ichannel in range(obs_ndim):
            b[:, ichannel] = \
                lstsq(h.reshape((obs_ndim, -1, 1 + lag))[ichannel, :], y.reshape((-1, obs_ndim))[:, ichannel])[0]

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
                     chol=chol,
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

    # print all options
    # for k, v in options.items():
    #     print(k, v)

    inference = postprocess(infer(model_fit, options))
    return inference, options


def fill_options(options):
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
    options['verbose'] = options.get('verbose', False)  # detailed output
    options['niter'] = options.get('niter', 200)  # max # of iteration
    options['learn_post'] = options.get('learn_post', True)  # optimize posterior
    options['learn_param'] = options.get('learn_param', True)  # optimize loading and regression
    options['learn_hyper'] = options.get('learn_hyper', True)
    options['nhyper'] = options.get('nhyper', 5)  # optimize hyperparams every # iterations
    options['decay'] = options.get('decay', 0)  # decay rate of the second moment of gradient. TODO: move to optimizers
    # options['sigma_factor'] = options.get('sigma_factor',
    #                                       5)  # multiplicative step length for optimizing sigma. TODO: remove
    # options['omega_factor'] = options.get('omega_factor',
    #                                       5)  # multiplicative step length for optimizing omega. TODO: remove
    options['hessian'] = options.get('hessian', True)  # use Hessian in M-step
    options['adjust_hessian'] = options.get('adjust_hessian', False)  # regular Hessian by gradient
    options['learning_rate'] = options.get('learning_rate', 0.001)  # learning rate
    options['method'] = options.get('method', 'VB')  # method of estimate, 'VB' or 'MAP'
    options['post_prediction'] = options.get('post_prediction', True)  # use posterior variance in predicted firing rate
    options['backtrack'] = options.get('backtrack', False)  # recover from decreased ELBO. TODO: remove
    options['e_niter'] = options.get('e_niter', 1)  # max # of estep loop
    options['m_niter'] = options.get('m_niter', 1)  # max # of mstep loop
    options['update_bound'] = options.get('update_bound', 1.0)
    options['subsample_size'] = options.get('subsample_size', None)
    options['hyper_obj'] = options.get('hyper_obj', 'ELBO')
    options['Adam'] = options.get('Adam', False)
    options['gp_noise'] = options.get('gp_noise', 1e-3)
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
    ntrial = model_fit['mu'].shape[0]
    chol = model_fit['chol']
    dyn_ndim, nbin, rank = chol.shape
    w = model_fit['w']
    eyer = identity(rank)
    L = empty((ntrial, dyn_ndim, nbin, rank))
    for trial in range(ntrial):
        for dyn_dim in range(dyn_ndim):
            G = chol[dyn_dim, :, :]
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
    frate = np.exp(eta + 0.5 * v.reshape((-1, dyn_ndim)) @ (a ** 2)) if v is not None else np.exp(eta)
    yhat = frate.reshape(y.shape)
    return yhat
