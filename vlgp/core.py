"""Module that does inference"""
import timeit
import warnings

import numpy as np
from numpy import identity, einsum, trace, inner, empty, inf, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, arange, sum, copyto, ones
from numpy.core.umath import sqrt, PINF, log, exp
from numpy.linalg import slogdet, LinAlgError
from scipy import stats
from scipy.linalg import lstsq, eigh, solve, svd, norm
from scipy.stats import spearmanr
import sklearn.decomposition.factor_analysis

from .hyper import learngp
from .math import ichol_gauss, subspace, sexp
from .optimizer import AdamOptimizer
from .util import add_constant, rotate, lagmat


def elbo(obj):
    """Evidence Lower BOund
    Args:
        obj: inference object

    Returns:
        lb: lower bound
        ll: log-likelihood
    """
    obs_ndim, ntrial, nbin, lag1 = obj['h'].shape  # neuron, trial, time, lag1
    dyn_ndim, _, rank = obj['chol'].shape  # latent, time, rank

    eye_rank = identity(rank)

    y = obj['y'].reshape((-1, obs_ndim))  # concatenate trials
    h = obj['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials
    obs_types = obj['channel']

    chol = obj['chol']

    mu = obj['mu'].reshape((-1, dyn_ndim))
    v = obj['v'].reshape((-1, dyn_ndim))

    a = obj['a']
    b = obj['b']
    noise = obj['noise']

    spike = obs_types == 'spike'
    lfp = obs_types == 'lfp'

    # eta = mu @ a + einsum('ijk, ki -> ji', h.reshape((obs_ndim, nbin * ntrial, lag1)), b)
    # lam = sexp(eta + 0.5 * v @ (a ** 2))

    lam, eta = firing_rate(mu, v, a, b, h.reshape((obs_ndim, nbin * ntrial, lag1)))

    llspike = sum(y[:, spike] * eta[:, spike] - lam[:, spike])  # verified by predict()

    # noinspection PyTypeChecker
    lllfp = - 0.5 * sum(((y[:, lfp] - eta[:, lfp]) ** 2 + v @ (a[:, lfp] ** 2)) / noise[lfp] + log(noise[lfp]))

    ll = llspike + lllfp

    lb = ll

    for itrial in range(ntrial):
        mu = obj['mu'][itrial, :]
        w = obj['w'][itrial, :]
        for l in range(dyn_ndim):
            G = chol[l, :]
            GtWG = G.T @ (w[:, [l]] * G)
            tmp = GtWG @ solve(eye_rank + GtWG, GtWG, sym_pos=True)  # expected to be nonsingular
            # TODO: The mu^T K^{-1} mu below needs a good approximate than least squares.
            G_mldiv_mu = lstsq(G, mu[:, l])[0]
            mu_Kinv_mu = inner(G_mldiv_mu, G_mldiv_mu)
            # K = G @ G.T
            # mu_Kinv_mu = mu[:, l] @ spilu(csc_matrix(K)).solve(mu[:, l])
            tr = nbin - trace(GtWG) + trace(tmp)
            lndet = slogdet(eye_rank - GtWG + tmp)[1]

            lb += -0.5 * mu_Kinv_mu - 0.5 * tr + 0.5 * lndet + 0.5 * nbin

    return lb, ll


def firing_rate(mu, v, a, b, h):
    # eta_b = np.vstack(h_row @ b_col for h_row, b_col in zip(h, b.T)).T  # slower way
    eta = mu @ a + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) x (lag, neuron) -> (time, neuron)
    lam = sexp(eta + 0.5 * v @ (a ** 2))
    return lam, eta


def accumulate(acc, grad, b=1):
    """Accumulate gradient for Hessian adjustment

    Args:
        acc: accumulation matrix
        grad: new gradient
        b: expoential decay

    Returns:
        sum of squared gradients
    """
    if b < 1:
        return b * acc + (1 - b) * grad ** 2
    else:
        return acc + grad ** 2


def estep(obj, options):
    """Posterior step
    Args:
        obj: inference object
        options: controlling inference

    Returns:
        inference object
    """
    obs_ndim, ntrial, nbin, lag1 = obj['h'].shape  # neuron, trial, time, lag
    dyn_ndim, _, rank = obj['chol'].shape  # latent, time, rank

    obs_types = obj['channel']

    chol = obj['chol']

    a = obj['a']
    b = obj['b']
    noise = obj['noise']

    dmu_acc = options['dmu_acc']
    decay = options['decay']
    adjust_hessian = options['adjhess']
    eps = options['eps']

    spike = obs_types == 'spike'
    lfp = obs_types == 'lfp'

    eyer = identity(rank)
    residual = empty((nbin, obs_ndim), dtype=float)
    U = empty((nbin, obs_ndim), dtype=float)

    for trial in range(ntrial):
        # trial slices
        y = obj['y'][trial, :]
        h = obj['h'][:, trial, :, :]
        mu = obj['mu'][trial, :]
        w = obj['w'][trial, :]
        v = obj['v'][trial, :]

        hb = einsum('ijk, ki -> ji', h, b)
        eta = mu @ a + hb
        lam = sexp(eta + 0.5 * v @ (a ** 2))
        for dyn_dim in range(dyn_ndim):
            # lam, eta = firing_rate(mu, v, a, b, h)
            G = chol[dyn_dim, :, :]

            residual[:, spike] = y[:, spike] - lam[:, spike]  # residuals of Poisson observations
            residual[:, lfp] = (y[:, lfp] - eta[:, lfp]) / noise[lfp]  # residuals of Gaussian observations

            grad_mu_resid = (y[:, spike] - lam[:, spike]) @ a[dyn_dim, spike] + \
                            ((y[:, lfp] - eta[:, lfp]) / noise[lfp]) @ a[dyn_dim, lfp]
            # inner loop
            optimizer = options['optimizer_mu'][trial, dyn_dim]
            old_slice = mu[:, dyn_dim].copy()
            for _ in range(options['inner_niter']):
                grad_mu = grad_mu_resid  # - lstsq(G.T, lstsq(G, old_slice)[0])[0]
                # grad_mu = grad_mu_resid
                dmu_acc[trial, :, dyn_dim] = accumulate(dmu_acc[trial, :, dyn_dim], grad_mu, decay)

                if adjust_hessian:
                    wadj = (w[:, [dyn_dim]] + sqrt(eps + dmu_acc[trial, :, [dyn_dim]]))  # adjusted Hessian
                else:
                    wadj = w[:, [dyn_dim]]  # keep dimension
                GtWG = G.T @ (wadj * G)

                u = G @ (G.T @ (residual @ a[dyn_dim, :])) - old_slice
                delta_mu = u - G @ ((wadj * G).T @ u) + \
                           G @ (GtWG @ solve(eyer + GtWG, (wadj * G).T @ u, sym_pos=True))

                if options['Adam']:
                    delta_mu = optimizer.update(delta_mu)
                new_slice = old_slice + delta_mu

                # if np.allclose(old_slice, new_slice):
                #     break
                old_slice = new_slice

            mu[:, dyn_dim] = old_slice

        eta = mu @ a + hb
        lam = sexp(eta + 0.5 * v @ (a ** 2))
        # lam, eta = firing_rate(mu, v, a, b, h)
        U[:, spike] = lam[:, spike]
        U[:, lfp] = 1 / noise[lfp]
        w[:] = U @ (a.T ** 2)
        if options['method'] == 'VB':
            for dyn_dim in range(dyn_ndim):
                G = chol[dyn_dim, :, :]
                GtWG = G.T @ (w[:, [dyn_dim]] * G)
                v[:, dyn_dim] = (G * (G - G @ GtWG + G @ (GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)))).sum(
                    axis=1)

    # center over all trials if not only infer posterior
    if options['learn_param']:
        shape = obj['mu'].shape
        mu_over_trials = obj['mu'].reshape((-1, dyn_ndim))
        mean_over_trials = mu_over_trials.mean(axis=0)
        obj['b'][0, :] += mean_over_trials @ obj['a']  # compensate bias
        mu_over_trials -= mean_over_trials
        obj['mu'] = mu_over_trials.reshape(shape)


def mstep(obj, options):
    """Parameter step
    Args:
        obj: inference object
        options: optional arguments controlling inference

    Returns:
        inference object
    """
    obs_ndim, ntrial, nbin, lag1 = obj['h'].shape  # neuron, trial, time, lag + 1
    dyn_ndim, _, rank = obj['chol'].shape  # latent, time, rank

    y = obj['y'].reshape((-1, obs_ndim))  # concatenate trials
    h = obj['h'].reshape((obs_ndim, -1, lag1))  # concatenate trials
    obs_types = obj['channel']

    mu = obj['mu'].reshape((-1, dyn_ndim))
    v = obj['v'].reshape((-1, dyn_ndim))

    a = obj['a']
    b = obj['b']

    decay = options['decay']
    adjust_hessian = options['adjhess']
    eps = options['eps']
    da_acc = options['da_acc']
    db_acc = options['db_acc']

    # TODO: # of neuron is much larger than L and p. It would better not looping on neuron
    lam, eta = firing_rate(mu, v, a, b, h)
    for obs_dim in range(obs_ndim):
        optimizer_a = options['optimizer_a'][obs_dim]
        optimizer_b = options['optimizer_b'][obs_dim]

        # eta = mu @ a + einsum('ijk, ki -> ji', h, b)
        # lam = sexp(eta + 0.5 * v @ (a ** 2))
        if obs_types[obs_dim] == 'spike':
            # loading
            va = v * a[:, obs_dim]  # (ntime, nlatent)
            wv = diag(lam[:, obs_dim] @ v)

            # inner loop
            old_slice = a[:, obs_dim].copy()
            for _ in range(options['inner_niter']):
                grad_a = mu.T @ y[:, obs_dim] - (mu + va).T @ lam[:, obs_dim]
                # grad_a = mu.T.dot(y[:, train] - lam[:, train])
                da_acc[:, obs_dim] = accumulate(da_acc[:, obs_dim], grad_a, decay)

                if options['hessian']:
                    neghess_a = (mu + va).T @ diag(lam[:, obs_dim]) @ (mu + va) + wv
                    # neghess_a = mu.T.dot(lam[:, train, newaxis] * mu)

                    if adjust_hessian:
                        delta_a = solve(neghess_a + diag(sqrt(eps + da_acc[:, obs_dim])), grad_a, sym_pos=True)
                    else:
                        try:
                            delta_a = solve(neghess_a, grad_a, sym_pos=True)
                        except LinAlgError:
                            print('singular Hessian a')
                            delta_a = grad_a
                else:
                    delta_a = grad_a

                if options['Adam']:
                    delta_a = optimizer_a.update(delta_a)
                new_slice = old_slice + delta_a
                # if np.allclose(old_slice, new_slice):
                #     break
                old_slice = new_slice
            a[:, obs_dim] = old_slice

            # bias
            old_slice = b[:, obs_dim].copy()
            for _ in range(options['inner_niter']):
                grad_b = h[obs_dim, :].T @ (y[:, obs_dim] - lam[:, obs_dim])
                db_acc[:, obs_dim] = accumulate(db_acc[:, obs_dim], grad_b, decay)

                if options['hessian']:
                    neghess_b = h[obs_dim, :].T @ diag(lam[:, obs_dim]) @ h[obs_dim, :]
                    # TODO: inactive neurons never fire across all trials which may cause zero Hessian
                    if adjust_hessian:
                        delta_b = solve(neghess_b + diag(sqrt(eps + db_acc[:, obs_dim])), grad_b, sym_pos=True)
                    else:
                        try:
                            delta_b = solve(neghess_b, grad_b, sym_pos=True)
                        except LinAlgError:
                            print('singular Hessian b')
                            delta_b = grad_b
                else:
                    delta_b = grad_b

                if options['Adam']:
                    delta_b = optimizer_b.update(delta_b)
                new_slice = old_slice + delta_b
                # if np.allclose(old_slice, new_slice):
                #     break
                old_slice = new_slice
            b[:, obs_dim] = old_slice
        elif obs_types[obs_dim] == 'lfp':
            # a's least squares solution for Gaussian channel
            # (m'm + diag(j'v))^-1 m'(y - Hb)
            a[:, obs_dim] = solve(mu.T @ mu + diag(sum(v, axis=0)),
                                   mu.T @ (y[:, obs_dim] - h[obs_dim, :] @ b[:, obs_dim]), sym_pos=True)

            # b's least squares solution for Gaussian channel
            # (H'H)^-1 H'(y - ma)
            b[:, obs_dim] = solve(h[obs_dim, :].T @ h[obs_dim, :],
                                   h[obs_dim, :].T @ (y[:, obs_dim] - mu @ a[:, obs_dim]), sym_pos=True)
        else:
            raise ValueError('Unsupported channel')
    lam, eta = firing_rate(mu, v, a, b, h)
    obj['noise'] = var(y - eta, axis=0, ddof=0)  # MLE

    # normalize loading by latent and rescale latent
    if options['learn_post']:
        scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
        a /= scale
        mu *= scale.squeeze()  # compensate latent
        obj['mu'] = mu.reshape(obj['mu'].shape)
        # noinspection PyTupleAssignmentBalance
        # U, s, Vh = svd(a, full_matrices=False)
        # obj['mu'] = np.reshape(mu @ a @ Vh.T, (ntrial, nbin, dyn_ndim))
        # a[:] = Vh


def fill_default_args(options):
    """Fill default values of controlling arguments if missing
    Args:
        options: optional arguments controlling inference

    Returns:
        valid arguments
    """
    options['verbose'] = options.get('verbose', False)
    options['niter'] = options.get('niter', 50)
    # options['infer'] = options.get('infer', 'both')
    options['learn_post'] = options.get('learn_post', True)
    options['learn_param'] = options.get('learn_param', True)
    options['learn_sigma'] = options.get('learn_sigma', False)
    options['learn_omega'] = options.get('learn_omega', False)
    options['tol'] = options.get('tol', 1e-5)
    options['eps'] = options.get('eps', 1e-8)
    options['nhyper'] = options.get('nhyper', 5)
    options['decay'] = options.get('decay', 0)
    options['sigma_factor'] = options.get('sigma_factor', 5)
    options['omega_factor'] = options.get('omega_factor', 5)
    options['hessian'] = options.get('hessian', False)
    options['moreparam'] = options.get('moreparam', False)
    options['adjhess'] = options.get('adjhess', True)
    options['learning_rate'] = options.get('learning_rate', 0.001)
    options['method'] = options.get('method', 'VB')
    options['post_prediction'] = options.get('post_prediction', True)
    options['backtrack'] = options.get('backtrack', True)
    options['inner_niter'] = options.get('inner_niter', 1)
    return options


def infer(obj, options):
    """Main inference procedure
    Args:
        obj: inference object
        options: optional arguments controlling inference

    Returns:
        inference object
    """

    # for backtracking
    good_mu = obj['mu'].copy()
    good_w = obj['w'].copy()
    good_v = obj['v'].copy()
    good_a = obj['a'].copy()
    good_b = obj['b'].copy()
    good_noise = obj['noise'].copy()
    good_sigma = obj['sigma'].copy()
    good_omega = obj['omega'].copy()

    stat = empty(options['niter'], dtype=object)
    lb = zeros(options['niter'], dtype=float)
    ll = zeros(options['niter'], dtype=float)
    elapsed = zeros((options['niter'], 3), dtype=float)
    loading_angle = zeros(options['niter'], dtype=float)
    latent_angle = zeros(options['niter'], dtype=float)
    latent_corr = zeros((options['niter'], obj['mu'].shape[-1]), dtype=float)
    dyn_ndim, ntime, rank = obj['chol'].shape

    x = obj.get('x')
    alpha = obj.get('alpha')

    # iteration 0
    # lb[0], ll[0] = elbo(obj)
    lb[0], ll[0] = np.finfo(float).min, np.finfo(float).min
    if alpha is not None:
        loading_angle[0] = subspace(alpha.T, obj['a'].T)
    if x is not None:
        rotated = empty_like(x, dtype=float)
        # rotate trial by trial
        for itrial in range(x.shape[0]):
            rotated[itrial, :] = rotate(add_constant(obj['mu'][itrial, :]), x[itrial, :])
        latent_angle[0] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
        rho, _ = spearmanr(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
        latent_corr[0] = rho[np.arange(x.shape[-1]), np.arange(x.shape[-1]) + x.shape[-1]]

    #
    iiter = 1
    converged = False
    stop = False
    infer_tick = timeit.default_timer()

    logging_counter = 0

    if options['verbose']:
        print('\nInference starts')

    while not stop and iiter < options['niter']:
        iter_tick = timeit.default_timer()

        # options['learning_rate'] = 2 / (1 + 1.002 ** (iiter - 1))

        # infer posterior
        post_tick = timeit.default_timer()
        if options['learn_post']:
            estep(obj, options)
        # elbo(obj)
        post_tock = timeit.default_timer()
        elapsed[iiter, 0] = post_tock - post_tick

        # Calculate angle between latent subspace if true latent is given.
        if x is not None:
            for itrial in range(x.shape[0]):
                rotated[itrial, :] = rotate(add_constant(obj['mu'][itrial, :]), x[itrial, :])
            latent_angle[iiter] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
            rho, _ = spearmanr(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))
            latent_corr[iiter] = rho[np.arange(x.shape[-1]), np.arange(x.shape[-1]) + x.shape[-1]]

        # infer parameter
        param_tick = timeit.default_timer()
        if options['learn_param']:
            mstep(obj, options)
        param_tock = timeit.default_timer()
        elapsed[iiter, 1] = param_tock - param_tick

        # Calculate angle between loading subspace if true loading is given.
        if alpha is not None:
            loading_angle[iiter] = subspace(alpha.T, obj['a'].T)

        lb[iiter], ll[iiter] = 0, 0
        decreased = lb[iiter] < lb[iiter - 1]
        converged = np.allclose(obj['mu'], good_mu)

        if decreased:
            # if options['verbose']:
            #     print('\nELBO decreased.')
            if options['backtrack']:
                copyto(obj['mu'], good_mu)
                copyto(obj['w'], good_w)
                copyto(obj['v'], good_v)
                copyto(obj['a'], good_a)
                copyto(obj['b'], good_b)
                copyto(obj['noise'], good_noise)
                lb[iiter] = lb[iiter - 1]
            else:
                copyto(good_mu, obj['mu'])
                copyto(good_w, obj['w'])
                copyto(good_v, obj['v'])
                copyto(good_a, obj['a'])
                copyto(good_b, obj['b'])
                copyto(good_noise, obj['noise'])
        else:
            copyto(good_mu, obj['mu'])
            copyto(good_w, obj['w'])
            copyto(good_v, obj['v'])
            copyto(good_a, obj['a'])
            copyto(good_b, obj['b'])
            copyto(good_noise, obj['noise'])

        # converged = converged  # or abs(lb[iiter] - lb[iiter - 1]) < options['tol'] * abs(lb[iiter - 1])
        # stop = converged or decreased
        stop = converged

        if iiter % options['nhyper'] == 0 and (options['learn_sigma'] or options['learn_omega']):
            gp = learngp(obj, **options)
            copyto(obj['sigma'], gp[0])
            copyto(obj['omega'], gp[1])
            for dyn_dim in range(dyn_ndim):
                obj['chol'][dyn_dim, :] = ichol_gauss(ntime, obj['omega'][dyn_dim], rank) * obj['sigma'][dyn_dim]
            lbhyper, _ = elbo(obj)
            if lbhyper < lb[iiter]:
                copyto(obj['sigma'], good_sigma)
                copyto(obj['omega'], good_omega)
                for dyn_dim in range(dyn_ndim):
                    obj['chol'][dyn_dim, :] = ichol_gauss(ntime, obj['omega'][dyn_dim], rank) * obj['sigma'][dyn_dim]
            else:
                copyto(good_sigma, obj['sigma'])
                copyto(good_omega, obj['omega'])

        iter_tock = timeit.default_timer()
        elapsed[iiter, 2] = iter_tock - iter_tick

        # statistics of current iteration
        stat[iiter] = {}
        stat[iiter]['Elapsed Post'] = elapsed[iiter, 0]
        stat[iiter]['Elapsed Param'] = elapsed[iiter, 1]
        stat[iiter]['Elapsed Total'] = elapsed[iiter, 2]
        stat[iiter]['ELBO'] = lb[iiter]
        stat[iiter]['LL'] = ll[iiter]
        stat[iiter]['sigma'] = good_sigma
        stat[iiter]['omega'] = good_omega

        # TODO: change stat to OrderedDict
        if options['verbose'] and iiter == 2 ** logging_counter:
            print('\n[{}]'.format(iiter))
            for k in sorted(stat[iiter]):
                print('{}: {}'.format(k, stat[iiter][k]))
            logging_counter += 1

        iiter += 1
    infer_tock = timeit.default_timer()

    if options['verbose']:
        print('\nInference ends')
        print('{} iterations, ELBO: {:.4f}, elapsed: {:.3f}, converged: {}\n'.format(iiter - 1,
                                                                                     lb[iiter - 1],
                                                                                     infer_tock - infer_tick,
                                                                                     converged))
    obj['ELBO'] = lb[:iiter]
    obj['Elapsed'] = elapsed[:iiter, :]
    obj['LoadingAngle'] = loading_angle[:iiter]
    obj['LatentAngle'] = latent_angle[:iiter]
    obj['RankCorr'] = latent_corr[:iiter]
    obj['LL'] = ll[:iiter]
    obj['stat'] = stat
    return obj


def fit(y, channel, sigma, omega, a=None, b=None, mu=None, x=None, alpha=None, beta=None, lag=0,
        rank=500, eps=1e-8, tol=1e-5, **kwargs):
    """Inference API
    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior timescale
        a:       initial value of loading
        b:       initial value of regression
        mu:      initial value of latent
        x:       optional true latent
        alpha:   optional true loading
        beta:    optional true regression
        lag:     autoregressive lag
        rank:    prior covariance rank
        niter: max number of iterations
        tol: relative tolerance for checking convergence
        adjhess: adjust Hessian by gradient
        decay: decay speed for the adjustment to Hessian
        verbose: display info in every iteration
        learn_post: optimize the posterior
        learn_param: optimize the parameters
        learn_sigma: optimize prior variance
        learn_omega: optimize prior timescale
        nhyper: optimize hyperparameters every nhyper iteration

    Returns:
        inference object
    """
    assert sigma.shape == omega.shape
    options = fill_default_args(kwargs)
    options['eps'] = eps
    options['tol'] = tol

    y = asarray(y)
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]

    channel = asarray(channel)
    ntrial, nbin, obs_ndim = y.shape
    dyn_ndim = sigma.shape[0]

    # make design matrix of regression
    h = empty((obs_ndim, ntrial, nbin, 1 + lag), dtype=float)
    for ichannel in range(obs_ndim):
        for itrial in range(ntrial):
            h[ichannel, itrial, :] = add_constant(lagmat(y[itrial, :, ichannel], lag=lag))

    # make Cholesky of prior
    chol = empty((dyn_ndim, nbin, rank), dtype=float)
    for dyn_dim in range(dyn_ndim):
        chol[dyn_dim, :] = ichol_gauss(nbin, omega[dyn_dim], rank) * sigma[dyn_dim]

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent
    if a is None and mu is None:
        fa = sklearn.decomposition.factor_analysis.FactorAnalysis(n_components=dyn_ndim, svd_method='lapack')
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
        # mu = np.reshape(mu @ a @ Vh.T, (ntrial, ntime, nlatent))
        # a[:] = Vh
    else:
        if mu is None:
            mu = lstsq(a.T, y.reshape((-1, obs_ndim)).T)[0].T.reshape((ntrial, nbin, dyn_ndim))
        elif a is None:
            a = lstsq(mu.reshape((-1, dyn_ndim)), y.reshape((-1, obs_ndim)))[0]

    # initialize square root of posterior covariance
    L = empty((ntrial, dyn_ndim, nbin, rank))

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

    obj = dict(y=y, channel=channel, h=h, sigma=sigma, omega=omega, chol=chol, mu=mu, w=w, v=v, L=L, a=a, b=b,
               noise=noise, x=x, alpha=alpha, beta=beta)

    options['dmu_acc'] = zeros_like(mu)
    options['da_acc'] = zeros_like(a)
    options['db_acc'] = zeros_like(b)

    options['optimizer_mu'] = empty((ntrial, dyn_ndim), dtype=object)
    options['optimizer_a'] = empty(obs_ndim, dtype=object)
    options['optimizer_b'] = empty(obs_ndim, dtype=object)

    for each in np.nditer(options['optimizer_mu'], flags=['refs_ok'], op_flags=['readwrite']):
        each[...] = AdamOptimizer(nbin, options['learning_rate'])
    for each in np.nditer(options['optimizer_a'], flags=['refs_ok'], op_flags=['readwrite']):
        each[...] = AdamOptimizer(dyn_ndim, options['learning_rate'])
    for each in np.nditer(options['optimizer_b'], flags=['refs_ok'], op_flags=['readwrite']):
        each[...] = AdamOptimizer(b.shape[0], options['learning_rate'])

    inference = postprocess(infer(obj, options))
    return inference, options


def seqfit(y, channel, sigma, omega, lag=0, rank=500, copy=False, **kwargs):
    """Sequential inference
    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior timescale
        lag:     autoregressive lag
        rank:    prior covariance rank
        copy:    False: start from last inference
        **kwargs: optional arguments controlling inference

    Returns:
        list of inference objects
    """
    assert sigma.shape == omega.shape
    kwargs = fill_default_args(**kwargs)

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]
    channel = asarray(channel)
    ntrial, ntime, nchannel = y.shape
    nlatent = sigma.shape[0]

    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    for ichannel in range(nchannel):
        for itrial in range(ntrial):
            h[ichannel, itrial, :] = add_constant(lagmat(y[itrial, :, ichannel], lag=lag))

    chol = empty((nlatent, ntime, rank), dtype=float)
    for ilatent in range(nlatent):
        chol[ilatent, :] = ichol_gauss(ntime, omega[ilatent], rank) * sigma[ilatent]

    # initialize posterior
    fa = sklearn.decomposition.factor_analysis.FactorAnalysis(n_components=nlatent, svd_method='lapack')
    mu = fa.fit_transform(y.reshape((-1, nchannel)))
    a = fa.components_

    # constrain loading and center latent
    anorm = norm(a, ord=inf, axis=1)
    mu -= mu.mean(axis=0)
    mu *= anorm
    a /= anorm[..., newaxis]
    mu = mu.reshape((ntrial, ntime, nlatent))
    # L = empty((ntrial, nlatent, ntime, rank))
    w = zeros((ntrial, ntime, nlatent))
    v = np.repeat(sigma[newaxis, ...], ntrial * ntime, axis=1).reshape((ntrial, ntime, nlatent))

    # initialize parameters
    b = empty((1 + lag, nchannel), dtype=float)
    for ichannel in range(nchannel):
        b[:, ichannel] = lstsq(h.reshape((nchannel, -1, 1 + lag))[ichannel, :], y.reshape((-1, nchannel))[:, ichannel])[
            0]

    noise = var(y.reshape((-1, nchannel)), axis=0, ddof=0)
    objs = []
    if kwargs['verbose']:
        print('\nSequential fit')
    for ilatent in range(nlatent):
        print('\n{} latent(s)'.format(ilatent + 1))
        if copy:
            obj = dict(y=y, channel=channel, h=h, sigma=sigma[:ilatent + 1].copy(), omega=omega[:ilatent + 1].copy(),
                       chol=chol[:ilatent + 1, :].copy(), mu=mu[:, :, :ilatent + 1].copy(),
                       w=w[:, :, :ilatent + 1].copy(), v=v[:, :, :ilatent + 1].copy(), a=a[:ilatent + 1, :].copy(),
                       b=b.copy(), noise=noise.copy())
        else:
            obj = dict(y=y, channel=channel, h=h, sigma=sigma[:ilatent + 1], omega=omega[:ilatent + 1],
                       chol=chol[:ilatent + 1, :], mu=mu[:, :, :ilatent + 1], w=w[:, :, :ilatent + 1],
                       v=v[:, :, :ilatent + 1], a=a[:ilatent + 1, :], b=b, noise=noise)

        kwargs['dmu_acc'] = zeros_like(mu[:, :, :ilatent + 1])
        kwargs['da_acc'] = zeros_like(a[:ilatent + 1, :])
        kwargs['db_acc'] = zeros_like(b)

        objs.append(postprocess(infer(obj, **kwargs)))

    return objs


def leave_one_out(trial, model, **kwargs):
    """Leave-one-out prediction
    Args:
        trial: trial to predict
        model: fitted model
        kwargs: optional arguments controlling inference

    Returns:
        trial with prediction
    """
    kwargs = fill_default_args(**kwargs)
    y = trial['y']
    h = trial['h']
    channel = model['channel']
    yhat = trial['yhat']
    ntrial, ntime, nchannel = y.shape
    nlatent = model['mu'].shape[-1]

    a = model['a']
    b = model['b']

    for ichannel in range(nchannel):
        included = arange(nchannel) != ichannel
        ytrain = y[:, :, included]
        htrain = h[included, :]
        htest = h[ichannel, :]

        obj = {'y': ytrain, 'h': htrain, 'channel': channel[included]}

        # initialize posterior
        if trial['mu0'] is None:
            mu = lstsq(a.T, y.reshape((-1, nchannel)).T)[0].T.reshape((ntrial, ntime, nlatent))
        else:
            mu = trial['mu0']
        obj['mu'] = mu
        obj['sigma'] = model['sigma'].copy()
        obj['omega'] = model['omega'].copy()
        obj['chol'] = model['chol'].copy()
        # obj['L'] = zeros()
        obj['w'] = zeros_like(mu)
        obj['v'] = np.repeat(obj['sigma'][newaxis, ...], ntrial * ntime, axis=1).reshape((ntrial, ntime, nlatent))

        # set parameters
        obj['a'] = model['a'][:, included]
        obj['b'] = model['b'][:, included]
        obj['noise'] = model['noise'][included]

        # kwargs['infer'] = 'posterior'
        kwargs['learn_post'] = True
        kwargs['learn_param'] = False
        kwargs['learn_sigma'] = False
        kwargs['learn_omega'] = False

        obj = infer(obj, **kwargs)
        eta = obj['mu'].reshape((-1, nlatent)) @ a[:, ichannel] + htest.reshape((ntime * ntrial, -1)) @ b[:, ichannel]
        if channel[ichannel] == 'spike':
            if kwargs['post_prediction']:
                yhat[:, :, ichannel] = exp(
                    eta + 0.5 * obj['v'].reshape((-1, nlatent)) @ (a[:, ichannel] ** 2)).reshape(
                    (yhat.shape[0], yhat.shape[1]))
            else:
                yhat[:, :, ichannel] = exp(eta).reshape((yhat.shape[0], yhat.shape[1]))
        else:
            yhat[:, :, ichannel] = eta.reshape((yhat.shape[0], yhat.shape[1]))

    return trial


def cv(y, channel, sigma, omega, a0=None, mu0=None, lag=0, rank=500, **kwargs):
    """Cross-validation
    Do leave-one-out prediction to all trials. Use one trial as test and the rest as training each time.

    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior time scale
        a0:      initial loading
        mu0:     initial latent
        lag:     autoregressive lag
        rank:    prior covariance rank
        **kwargs: optional arguments controlling inference

    Returns:
        prediction of all neurons
    """
    kwargs = fill_default_args(**kwargs)
    assert sigma.shape == omega.shape

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]
    channel = asarray(channel)
    ntrial, ntime, nchannel = y.shape

    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    for ichannel in range(nchannel):
        for itrial in range(ntrial):
            h[ichannel, itrial, :] = add_constant(lagmat(y[itrial, :, ichannel], lag=lag))

    yhat = empty_like(y, dtype=float)
    # do leave-one-out trial by trial
    for itrial in range(ntrial):
        test_trial = {'y': y[itrial, :][newaxis, ...], 'h': h[:, itrial, :, :][:, newaxis, :, :],
                      'yhat': yhat[itrial, :][newaxis, ...],
                      'mu0': mu0[itrial, :][newaxis, ...] if mu0 is not None else None}
        itrain = arange(ntrial) != itrial
        model, _ = fit(y[itrain, :], channel, sigma, omega, x=None, a0=a0,
                       mu0=mu0[itrain, :] if mu0 is not None else None,
                       alpha=None, beta=None,
                       lag=lag, rank=rank, **kwargs)
        kwargs['verbose'] = False
        kwargs['dmu_acc'] = zeros_like(model['mu'])
        kwargs['da_acc'] = zeros_like(model['a'])
        kwargs['db_acc'] = zeros_like(model['b'])
        leave_one_out(test_trial, model, **kwargs)
    ll = stats.poisson.logpmf(y.ravel(), yhat.ravel()).reshape(y.shape)
    prediction = {'y': y, 'yhat': yhat, 'LL': ll}
    return prediction


def postprocess(obj):
    """Remove intermediate and empty variables, and compute decomposition of posterior covariance
    Args:
        obj: raw inference

    Returns:
        infernece object
    """
    ntrial = obj['mu'].shape[0]
    chol = obj['chol']
    nlatent, ntime, rank = chol.shape
    w = obj['w']
    eyer = identity(rank)
    L = empty((ntrial, nlatent, ntime, rank))
    for itrial in range(ntrial):
        for ilatent in range(nlatent):
            G = chol[ilatent, :, :]
            GtWG = G.T @ (w[itrial, :, [ilatent]] * G)
            try:
                tmp = eyer - GtWG + GtWG @ solve(eyer + GtWG, GtWG, sym_pos=True)  # A should be PD but numerically not
            except LinAlgError:
                warnings.warn('Singular matrix. Use least squares instead.')
                tmp = eyer - GtWG + GtWG @ lstsq(eyer + GtWG, GtWG)[0]  # least squares
            eigval, eigvec = eigh(tmp)
            eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
            L[itrial, ilatent, :] = G @ (eigvec @ diag(sqrt(eigval)))
    obj['L'] = L
    keys = list(obj.keys())
    for key in keys:
        if obj.get(key, None) is None:
            obj.pop(key, None)
    obj.pop('h', None)
    obj.pop('stat', None)
    return obj


def predict(y, x, a, b, v=None):
    """
    Predict firing rate
    Args:
        y: spike trains
        x: latent
        a: loading
        b: regression
        v: posterior variance

    Returns:
        yhat: predicted firing rate
    """
    ntrial, ntime, ntrain = y.shape
    nlatent = x.shape[-1]
    lag = b.shape[0] - 1

    # regression (h dot b) part
    reg = empty_like(y)
    for itrain in range(ntrain):
        for itrial in range(ntrial):
            h = add_constant(lagmat(y[itrial, :, itrain], lag=lag))
            reg[itrial, :, itrain] = h @ b[:, itrain]
    eta = x.reshape((-1, nlatent)) @ a + reg.reshape((-1, ntrain))
    lam = np.exp(eta + 0.5 * v.reshape((-1, nlatent)) @ (a ** 2)) if v is not None else np.exp(eta)
    yhat = lam.reshape(y.shape)
    return yhat
