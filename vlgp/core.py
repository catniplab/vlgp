"""
Inference
"""
import timeit
import warnings

import numpy as np
from numpy import identity, einsum, trace, inner, empty, inf, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, arange, sum, copyto, ones
from numpy.core.umath import sqrt, PINF, log, exp
from numpy.linalg import norm, slogdet, LinAlgError
from scipy import stats
from scipy.linalg import lstsq, eigh, solve
from sklearn.decomposition.factor_analysis import FactorAnalysis

from .math import ichol_gauss, subspace, sexp
from .util import add_constant, rotate, lagmat
from .hyper import learngp


def elbo(obj):
    """Evidence Lower BOund
    Args:
        obj: inference object

    Returns:
        lb: lower bound
        ll: log-likelihood
    """
    nchannel, ntrial, ntime, lag = obj['h'].shape  # neuron, trial, time, lag
    nlatent, _, rank = obj['chol'].shape  # latent, time, rank

    eyer = identity(rank)

    y = obj['y'].reshape((-1, nchannel))  # concatenate trials
    h = obj['h'].reshape((nchannel, -1, lag))  # concatenate trials
    channel = obj['channel']

    chol = obj['chol']

    mu = obj['mu'].reshape((-1, nlatent))
    v = obj['v'].reshape((-1, nlatent))

    a = obj['a']
    b = obj['b']
    noise = obj['noise']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eta = mu.dot(a) + einsum('ijk, ki -> ji', h.reshape((nchannel, ntime * ntrial, lag)), b)
    lam = sexp(eta + 0.5 * v.dot(a ** 2))

    llspike = sum(y[:, spike] * eta[:, spike] - lam[:, spike])  # verified by predict()

    lllfp = - 0.5 * sum(((y[:, lfp] - eta[:, lfp]) ** 2 + v.dot(a[:, lfp] ** 2)) / noise[lfp] + log(noise[lfp]))

    ll = llspike + lllfp

    lb = ll

    for itrial in range(ntrial):
        mu = obj['mu'][itrial, :]
        w = obj['w'][itrial, :]
        for l in range(nlatent):
            G = chol[l, :]
            GtWG = G.T.dot(w[:, l].reshape((ntime, 1)) * G)
            tmp = GtWG.dot(solve(eyer + GtWG, GtWG, sym_pos=True))  # expected to be nonsingular
            G_mldiv_mu = lstsq(G, mu[:, l])[0]
            tr = ntime - trace(GtWG) + trace(tmp)
            lndet = slogdet(eyer - GtWG + tmp)[1]

            lb += -0.5 * inner(G_mldiv_mu, G_mldiv_mu) - 0.5 * tr + 0.5 * lndet + 0.5 * ntime

    return lb, ll


def accumulate(accu, grad, decay=1):
    """Accumulate gradient for Hessian adjustment

    Args:
        accu: accumulation matrix
        grad: new gradient
        decay: expoential decay

    Returns:
        sum of squared gradients
    """
    if decay < 1:
        return decay * accu + (1 - decay) * grad ** 2
    else:
        return accu + grad ** 2


def inferpost(obj, **kwargs):
    """Posterior step
    Args:
        obj: inference object
        **kwargs: optional arguments controlling inference

    Returns:
        inference object
    """
    nchannel, ntrial, ntime, lag = obj['h'].shape  # neuron, trial, time, lag
    nlatent, _, rank = obj['chol'].shape  # latent, time, rank

    channel = obj['channel']

    chol = obj['chol']

    a = obj['a']
    b = obj['b']
    noise = obj['noise']

    dmu_acc = kwargs['dmu_acc']
    decay = kwargs['decay']
    adjhess = kwargs['adjhess']
    eps = kwargs['eps']
    learning_rate = kwargs['learning_rate']

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eyer = identity(rank)
    resid = empty((ntime, nchannel), dtype=float)
    U = empty((ntime, nchannel), dtype=float)

    for itrial in range(ntrial):
        # trial-wise
        y = obj['y'][itrial, :]
        h = obj['h'][:, itrial, :, :]
        mu = obj['mu'][itrial, :]
        w = obj['w'][itrial, :]
        v = obj['v'][itrial, :]

        # eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) . (lag, neuron)
        # lam = exp((eta + 0.5 * v.dot(a ** 2)).clip(MIN_EXP, MAX_EXP))
        for ilatent in range(nlatent):
            eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)  # (neuron, time, lag) . (lag, neuron)
            lam = sexp(eta + 0.5 * v.dot(a ** 2))
            G = chol[ilatent, :, :]

            grad_mu = (y[:, spike] - lam[:, spike]).dot(a[ilatent, spike]) + \
                      ((y[:, lfp] - eta[:, lfp]) /
                       noise[lfp]).dot(a[ilatent, lfp]) - lstsq(G.T, lstsq(G, mu[:, ilatent])[0])[0]

            dmu_acc[itrial, :, ilatent] = accumulate(dmu_acc[itrial, :, ilatent], grad_mu, decay)

            if adjhess:
                wadj = (w[:, [ilatent]] + sqrt(eps + dmu_acc[itrial, :, ilatent])[..., newaxis])  # adjusted Hessian
            else:
                wadj = w[:, [ilatent]]  # keep dimension
            GtWG = G.T.dot(wadj * G)

            # update temporal variance
            # if not kwargs['MAP']:
            #     GtWGv = G.T.dot(w[:, [ilatent]] * G)
            #     v[:, ilatent] = (G * (G - G.dot(GtWGv) +
            #                           G.dot(GtWGv.dot(solve(eyer + GtWGv, GtWGv, sym_pos=True))))).sum(axis=1)

            resid[:, spike] = y[:, spike] - lam[:, spike]  # residuals of Poisson observations
            resid[:, lfp] = (y[:, lfp] - eta[:, lfp]) / noise[lfp]  # residuals of Gaussian observations

            u = G.dot(G.T.dot(resid.dot(a[ilatent, :]))) - mu[:, ilatent]
            delta_mu = u - G.dot((wadj * G).T.dot(u)) + \
                       G.dot(GtWG.dot(solve(eyer + GtWG, (wadj * G).T.dot(u), sym_pos=True)))

            mu[:, ilatent] += learning_rate * delta_mu
            # mu[:, ilatent] -= mean(mu[:, ilatent])
            # scale = norm(mu[:, ilatent], ord=inf)
            # mu[:, ilatent] /= scale

            # center over all trials if not only infer posterior
            if kwargs['learn_param']:
                shape = obj['mu'].shape
                mu_over_trials = obj['mu'].reshape((-1, nlatent))
                mean_over_trials = mu_over_trials.mean(axis=0)
                obj['b'][0, :] += mean_over_trials.dot(obj['a'])  # compensate bias
                mu_over_trials -= mean_over_trials
                obj['mu'] = mu_over_trials.reshape(shape)

        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        U[:, spike] = lam[:, spike]
        U[:, lfp] = 1 / noise[lfp]
        w[:] = U.dot(a.T ** 2)
        if not kwargs['MAP']:
            for ilatent in range(nlatent):
                G = chol[ilatent, :, :]
                GtWG = G.T.dot(w[:, [ilatent]] * G)
                v[:, ilatent] = (G * (G - G.dot(GtWG) + G.dot(GtWG.dot(solve(eyer + GtWG, GtWG, sym_pos=True))))).sum(
                    axis=1)


def inferparam(obj, **kwargs):
    """Parameter step
    Args:
        obj: inference object
        **kwargs: optional arguments controlling inference

    Returns:
        inference object
    """
    nchannel, ntrial, ntime, lag1 = obj['h'].shape  # neuron, trial, time, lag + 1
    nlatent, _, rank = obj['chol'].shape  # latent, time, rank

    y = obj['y'].reshape((-1, nchannel))  # concatenate trials
    h = obj['h'].reshape((nchannel, -1, lag1))  # concatenate trials
    channel = obj['channel']

    mu = obj['mu'].reshape((-1, nlatent))
    v = obj['v'].reshape((-1, nlatent))

    a = obj['a']
    b = obj['b']

    decay = kwargs['decay']
    adjhess = kwargs['adjhess']
    eps = kwargs['eps']
    da_acc = kwargs['da_acc']
    db_acc = kwargs['db_acc']
    learning_rate = kwargs['learning_rate']

    for ichannel in range(nchannel):
        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        if channel[ichannel] == 'spike':
            # loading
            va = v * a[:, ichannel]  # (ntime, nlatent)
            wv = diag(lam[:, ichannel].dot(v))
            grad_a = mu.T.dot(y[:, ichannel]) - (mu + va).T.dot(lam[:, ichannel])
            # grad_a = mu.T.dot(y[:, train] - lam[:, train])
            da_acc[:, ichannel] = accumulate(da_acc[:, ichannel], grad_a, decay)

            if kwargs['param_opt'] == 'GA':
                delta_a = grad_a / sqrt(eps + da_acc[:, ichannel])
            else:
                neghess_a = (mu + va).T.dot(lam[:, ichannel, newaxis] * (mu + va)) + wv
                # neghess_a = mu.T.dot(lam[:, train, newaxis] * mu)

                if adjhess:
                    delta_a = solve(neghess_a + diag(sqrt(eps + da_acc[:, ichannel])), grad_a, sym_pos=True)
                else:
                    try:
                        delta_a = solve(neghess_a, grad_a, sym_pos=True)
                    except LinAlgError:
                        delta_a = .0
            a[:, ichannel] += learning_rate * delta_a

            # bias
            grad_b = h[ichannel, :].T.dot(y[:, ichannel] - lam[:, ichannel])
            db_acc[:, ichannel] = accumulate(db_acc[:, ichannel], grad_b, decay)
            if kwargs['param_opt'] == 'GA':
                delta_b = grad_b / sqrt(eps + db_acc[:, ichannel])
            else:
                neghess_b = h[ichannel, :].T.dot(lam[:, ichannel, newaxis] * h[ichannel, :])
                # TODO: inactive neurons never fire across all trials which may cause zero Hessian
                if adjhess:
                    delta_b = solve(neghess_b + diag(sqrt(eps + db_acc[:, ichannel])), grad_b, sym_pos=True)
                else:
                    try:
                        delta_b = solve(neghess_b, grad_b, sym_pos=True)
                    except LinAlgError:
                        delta_b = .0

            b[:, ichannel] += learning_rate * delta_b
        elif channel[ichannel] == 'lfp':
            # a's least squares solution for Gaussian channel
            # (m'm + diag(j'v))^-1 m'(y - Hb)
            a[:, ichannel] = solve(mu.T.dot(mu) + diag(sum(v, axis=0)),
                                   mu.T.dot(y[:, ichannel] - h[ichannel, :].dot(b[:, ichannel])),
                                   sym_pos=True)

            # b's least squares solution for Gaussian channel
            # (H'H)^-1 H'(y - ma)
            b[:, ichannel] = solve(h[ichannel, :].T.dot(h[ichannel, :]),
                                   h[ichannel, :].T.dot(y[:, ichannel] - mu.dot(a[:, ichannel])), sym_pos=True)
        else:
            raise ValueError('Unsupported channel')
        obj['noise'] = var(y - eta, axis=0, ddof=0)  # MLE

    # normalize loading by latent and rescale latent
    if kwargs['learn_post']:
        scale = norm(a, ord=inf, axis=1)[..., newaxis]
        a /= scale
        mu *= scale.squeeze()  # compensate latent
        obj['mu'] = mu.reshape(obj['mu'].shape)


def fill_default_args(**kwargs):
    """Fill default values of controlling arguments if missing
    Args:
        **kwargs: optional arguments controlling inference

    Returns:
        valid arguments
    """
    kwargs['verbose'] = kwargs.get('verbose', False)
    kwargs['niter'] = kwargs.get('niter', 50)
    # kwargs['infer'] = kwargs.get('infer', 'both')
    kwargs['learn_post'] = kwargs.get('learn_post', True)
    kwargs['learn_param'] = kwargs.get('learn_param', True)
    kwargs['learn_sigma'] = kwargs.get('learn_sigma', False)
    kwargs['learn_omega'] = kwargs.get('learn_omega', False)
    kwargs['nadjhess'] = kwargs.get('nadjhess', 5)
    kwargs['tol'] = kwargs.get('tol', 1e-5)
    kwargs['eps'] = kwargs.get('eps', 1e-6)
    kwargs['nhyper'] = kwargs.get('nhyper', 5)
    kwargs['decay'] = kwargs.get('decay', 0)
    kwargs['sigma_factor'] = kwargs.get('sigma_factor', 5)
    kwargs['omega_factor'] = kwargs.get('omega_factor', 5)
    kwargs['param_opt'] = kwargs.get('param_opt', 'NR')
    kwargs['moreparam'] = kwargs.get('moreparam', False)
    kwargs['adjhess'] = kwargs.get('adjhess', True)
    kwargs['learning_rate'] = kwargs.get('learning_rate', 1.0)
    kwargs['MAP'] = kwargs.get('MAP', False)
    kwargs['post_prediction'] = kwargs.get('post_prediction', True)
    return kwargs


def infer(obj, fstat=None, **kwargs):
    """Main inference procedure
    Args:
        obj: inference object
        fstat: external function calculateing statistics at each iteration
        kwargs: optional arguments controlling inference

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

    stat = empty(kwargs['niter'], dtype=object)
    lb = zeros(kwargs['niter'], dtype=float)
    ll = zeros(kwargs['niter'], dtype=float)
    elapsed = zeros((kwargs['niter'], 3), dtype=float)
    loading_angle = zeros(kwargs['niter'], dtype=float)
    latent_angle = zeros(kwargs['niter'], dtype=float)
    nlatent, ntime, rank = obj['chol'].shape

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

    #
    iiter = 1
    converged = False
    stop = False
    infer_tick = timeit.default_timer()

    if kwargs['verbose']:
        print('\nInference starts')

    while not stop and iiter < kwargs['niter']:
        iter_tick = timeit.default_timer()

        # infer posterior
        post_tick = timeit.default_timer()
        if kwargs['learn_post']:
            inferpost(obj, **kwargs)
        elbo(obj)
        post_tock = timeit.default_timer()
        elapsed[iiter, 0] = post_tock - post_tick

        # Calculate angle between latent subspace if true latent is given.
        if x is not None:
            for itrial in range(x.shape[0]):
                rotated[itrial, :] = rotate(add_constant(obj['mu'][itrial, :]), x[itrial, :])
            latent_angle[iiter] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))

        # infer parameter
        param_tick = timeit.default_timer()
        if kwargs['learn_param']:
            inferparam(obj, **kwargs)
        param_tock = timeit.default_timer()
        elapsed[iiter, 1] = param_tock - param_tick

        # Calculate angle between loading subspace if true loading is given.
        if alpha is not None:
            loading_angle[iiter] = subspace(alpha.T, obj['a'].T)

        lb[iiter], ll[iiter] = elbo(obj)
        converged = abs(lb[iiter] - lb[iiter - 1]) < kwargs['tol'] * abs(lb[iiter - 1])
        decreased = lb[iiter] < lb[iiter - 1]
        stop = converged or decreased

        if decreased:
            if kwargs['verbose']:
                print('\nELBO decreased. Backtracking.')
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

        if iiter % kwargs['nhyper'] == 0 and (kwargs['learn_sigma'] or kwargs['learn_omega']):
            gp = learngp(obj, **kwargs)
            copyto(obj['sigma'], gp[0])
            copyto(obj['omega'], gp[1])
            for ilatent in range(nlatent):
                obj['chol'][ilatent, :] = ichol_gauss(ntime, obj['omega'][ilatent], rank) * obj['sigma'][ilatent]
            lbhyper, _ = elbo(obj)
            if lbhyper < lb[iiter]:
                copyto(obj['sigma'], good_sigma)
                copyto(obj['omega'], good_omega)
                for ilatent in range(nlatent):
                    obj['chol'][ilatent, :] = ichol_gauss(ntime, obj['omega'][ilatent], rank) * obj['sigma'][ilatent]
            else:
                copyto(good_sigma, obj['sigma'])
                copyto(good_omega, obj['omega'])

        iter_tock = timeit.default_timer()
        elapsed[iiter, 2] = iter_tock - iter_tick

        # statistics of current iteration
        stat[iiter] = fstat(obj) if fstat is not None else {}
        stat[iiter]['Elapsed Post'] = elapsed[iiter, 0]
        stat[iiter]['Elapsed Param'] = elapsed[iiter, 1]
        stat[iiter]['Elapsed Total'] = elapsed[iiter, 2]
        stat[iiter]['ELBO'] = lb[iiter]
        stat[iiter]['LL'] = ll[iiter]
        stat[iiter]['sigma'] = good_sigma
        stat[iiter]['omega'] = good_omega

        # TODO: change stat to OrderedDict
        if kwargs['verbose']:
            print('\n[{}]'.format(iiter))
            for k in sorted(stat[iiter]):
                print('{}: {}'.format(k, stat[iiter][k]))

        iiter += 1
    infer_tock = timeit.default_timer()

    if kwargs['verbose']:
        print('\nInference ends')
        print('{} iterations, ELBO: {:.4f}, elapsed: {:.2f}, converged: {}\n'.format(iiter - 1,
                                                                                     lb[iiter - 1],
                                                                                     infer_tock - infer_tick,
                                                                                     converged))
    obj['ELBO'] = lb[:iiter]
    obj['Elapsed'] = elapsed[:iiter, :]
    obj['LoadingAngle'] = loading_angle[:iiter]
    obj['LatentAngle'] = latent_angle[:iiter]
    obj['LL'] = ll[:iiter]
    obj['stat'] = stat
    return obj


def fit(y, channel, sigma, omega, a=None, b=None, mu=None, x=None, alpha=None, beta=None, lag=0,
        rank=500, **kwargs):
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
        **kwargs: optional arguments controlling inference

    Returns:
        inference object
    """
    assert sigma.shape == omega.shape
    kwargs = fill_default_args(**kwargs)

    y = asarray(y)
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]

    channel = asarray(channel)
    ntrial, ntime, nchannel = y.shape
    nlatent = sigma.shape[0]

    # make design matrix of regression
    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    for ichannel in range(nchannel):
        for itrial in range(ntrial):
            h[ichannel, itrial, :] = add_constant(lagmat(y[itrial, :, ichannel], lag=lag))

    # make Cholesky of prior
    chol = empty((nlatent, ntime, rank), dtype=float)
    for ilatent in range(nlatent):
        chol[ilatent, :] = ichol_gauss(ntime, omega[ilatent], rank) * sigma[ilatent]

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent
    if a is None and mu is None:
        fa = FactorAnalysis(n_components=nlatent, svd_method='lapack')
        mu = fa.fit_transform(y.reshape((-1, nchannel)))
        a = fa.components_

        # constrain loading and center latent
        mu -= mu.mean(axis=0)
        mu *= norm(a, ord=inf, axis=1)
        a /= norm(a, ord=inf, axis=1)[..., newaxis]
        mu = mu.reshape((ntrial, ntime, nlatent))
    else:
        if a is not None:
            mu = lstsq(a.T, y.reshape((-1, nchannel)).T)[0].T.reshape((ntrial, ntime, nlatent))
        elif mu is not None:
            a = lstsq(mu.reshape((-1, nlatent)), y.reshape((-1, nchannel)))[0]

    # initialize square root of posterior covariance
    L = empty((ntrial, nlatent, ntime, rank))

    # initialize bias and autoregression
    if b is None:
        b = empty((1 + lag, nchannel), dtype=float)
        for ichannel in range(nchannel):
            b[:, ichannel] = \
                lstsq(h.reshape((nchannel, -1, 1 + lag))[ichannel, :], y.reshape((-1, nchannel))[:, ichannel])[0]

    # initialize noises of guassian channels
    noise = var(y.reshape((-1, nchannel)), axis=0, ddof=0)

    # w and v
    w = 0 * ones((ntrial, ntime, nlatent), dtype=float)
    if kwargs['MAP']:
        v = 0 * ones((ntrial, ntime, nlatent), dtype=float)
    else:
        v = np.repeat(sigma[newaxis, ...], ntrial * ntime, axis=1).reshape((ntrial, ntime, nlatent))

    obj = dict(y=y, channel=channel, h=h, sigma=sigma, omega=omega, chol=chol, mu=mu, w=w, v=v, L=L, a=a, b=b,
               noise=noise, x=x, alpha=alpha, beta=beta)

    kwargs['dmu_acc'] = zeros_like(mu)
    kwargs['da_acc'] = zeros_like(a)
    kwargs['db_acc'] = zeros_like(b)

    inference = postprocess(infer(obj, **kwargs))
    return inference, kwargs


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
    fa = FactorAnalysis(n_components=nlatent, svd_method='lapack')
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
        eta = obj['mu'].reshape((-1, nlatent)).dot(a[:, ichannel]) + htest.reshape((ntime * ntrial, -1)).dot(
            b[:, ichannel])
        if channel[ichannel] == 'spike':
            if kwargs['post_prediction']:
                yhat[:, :, ichannel] = exp(eta + 0.5 * obj['v'].reshape((-1, nlatent)).dot(a[:, ichannel] ** 2)).reshape(
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
        model = fit(y[itrain, :], channel, sigma, omega, x=None, a0=a0, mu0=mu0[itrain, :] if mu0 is not None else None,
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
            GtWG = G.T.dot(w[itrial, :, ilatent].reshape((ntime, 1)) * G)
            try:
                tmp = eyer - GtWG + GtWG.dot(
                    solve(eyer + GtWG, GtWG, sym_pos=True))  # A should be PD but numerically not
            except LinAlgError:
                warnings.warn('Singular matrix. Use least squares instead.')
                tmp = eyer - GtWG + GtWG.dot(lstsq(eyer + GtWG, GtWG)[0])  # least squares
            eigval, eigvec = eigh(tmp)
            eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
            L[itrial, ilatent, :] = G.dot(eigvec.dot(diag(sqrt(eigval))))
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
            reg[itrial, :, itrain] = h.dot(b[:, itrain])
    eta = x.reshape((-1, nlatent)).dot(a) + reg.reshape((-1, ntrain))
    # eta = x.reshape((-1, nlatent)).dot(a) + b[0, :]
    lam = np.exp(eta + 0.5 * v.reshape((-1, nlatent)).dot(a ** 2)) if v is not None else np.exp(eta)
    yhat = lam.reshape(y.shape)
    return yhat
