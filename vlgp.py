"""
This file contains the functions used for inference.
"""
import timeit

import numpy as np
from numpy import identity, einsum, trace, inner, empty, mean, inf, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, arange, sum, copyto
from numpy.core.umath import sqrt, PINF, log, exp, NINF
from numpy.linalg import norm, slogdet, LinAlgError
from scipy import stats
from scipy.linalg import lstsq, eigh, solve
from sklearn.decomposition.factor_analysis import FactorAnalysis

from hyper import learngp
from mathf import ichol_gauss, subspace, sexp
from util import add_constant, rotate, lagmat


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

    llspike = sum(y[:, spike] * eta[:, spike] - lam[:, spike])

    lllfp = - 0.5 * sum(((y[:, lfp] - eta[:, lfp]) ** 2 + v.dot(a[:, lfp] ** 2)) / noise[lfp] + log(noise[lfp]))

    ll = llspike + lllfp

    lb = ll

    for itrial in range(ntrial):
        mu = obj['mu'][itrial, :]
        w = obj['w'][itrial, :]
        for l in range(nlatent):
            G = chol[l, :]
            GtWG = G.T.dot(w[:, l].reshape((ntime, 1)) * G)
            A = GtWG.dot(solve(eyer + GtWG, GtWG, sym_pos=True))
            G_mldiv_mu = lstsq(G, mu[:, l])[0]
            tr = ntime - trace(GtWG) + trace(A)
            lndet = slogdet(eyer - GtWG + A)[1]

            lb += -0.5 * inner(G_mldiv_mu, G_mldiv_mu) - 0.5 * tr + 0.5 * lndet + 0.5 * ntime

    return lb, ll


def accumulate(accu, grad, decay=0):
    """Accumulate gradient for Hessian adjustment

    Args:
        accu: accumulation matrix
        grad: new gradient
        decay: expoential decay

    Returns:

    """
    if decay > 0:
        return decay * accu + (1 - decay) * grad ** 2
    else:
        return accu + grad ** 2


def inferpost(obj, **kwargs):
    """Posterior step
    Args:
        obj: inference object
        **kwargs: optional arguments controlling inference

    Returns:

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

    spike = channel == 'spike'
    lfp = channel == 'lfp'

    eyer = identity(rank)
    res = empty((ntime, nchannel), dtype=float)
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
                wadj = (w[:, ilatent] + sqrt(eps + dmu_acc[itrial, :, ilatent])).reshape(
                        (ntime, 1))  # adjusted Hessian
            else:
                wadj = w[:, ilatent].reshape((ntime, 1))
            GtWG = G.T.dot(wadj * G)

            res[:, spike] = y[:, spike] - lam[:, spike]
            res[:, lfp] = (y[:, lfp] - eta[:, lfp]) / noise[lfp]

            u = G.dot(G.T.dot(res.dot(a[ilatent, :]))) - mu[:, ilatent]
            delta_mu = u - G.dot((wadj * G).T.dot(u)) + \
                       G.dot(GtWG.dot(solve(eyer + GtWG, (wadj * G).T.dot(u), sym_pos=True)))

            mu[:, ilatent] += delta_mu
            # mu[:, ilatent] -= mean(mu[:, ilatent])
            # scale = norm(mu[:, ilatent], ord=inf)
            # mu[:, ilatent] /= scale
            shape = obj['mu'].shape
            mu_over_trials = obj['mu'].reshape((-1, nlatent))
            mu_over_trials -= mu_over_trials.mean(axis=0)
            obj['mu'] = mu_over_trials.reshape(shape)

        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        U[:, spike] = lam[:, spike]
        U[:, lfp] = 1 / noise[lfp]
        w[:, :] = U.dot(a.T ** 2)

    # center over all trials


def inferparam(obj, **kwargs):
    """Parameter step
    Args:
        obj: inference object
        **kwargs: optional arguments controlling inference

    Returns:

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

    for ichannel in range(nchannel):
        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        if channel[ichannel] == 'spike':
            # loading
            va = v * a[:, ichannel]  # (ntime, nlatent)
            wv = diag(lam[:, ichannel].dot(v))
            grad_a = mu.T.dot(y[:, ichannel]) - (mu + va).T.dot(lam[:, ichannel])
            # grad_a = mu.T.dot(y[:, ichannel] - lam[:, ichannel])
            da_acc[:, ichannel] = accumulate(da_acc[:, ichannel], grad_a, decay)

            if kwargs['param_opt'] == 'GA':
                a[:, ichannel] += grad_a / sqrt(eps + da_acc[:, ichannel])
            else:
                neghess_a = (mu + va).T.dot(lam[:, ichannel, newaxis] * (mu + va)) + wv
                # neghess_a = mu.T.dot(lam[:, ichannel, newaxis] * mu)

                if adjhess:
                    delta_a = solve(neghess_a + diag(sqrt(eps + da_acc[:, ichannel])), grad_a, sym_pos=True)
                else:
                    delta_a = solve(neghess_a, grad_a, sym_pos=True)
                a[:, ichannel] += delta_a

            # bias
            grad_b = h[ichannel, :].T.dot(y[:, ichannel] - lam[:, ichannel])
            db_acc[:, ichannel] = accumulate(db_acc[:, ichannel], grad_b, decay)
            if kwargs['param_opt'] == 'GA':
                b[:, ichannel] += grad_b / sqrt(eps + db_acc[:, ichannel])
            else:
                neghess_b = h[ichannel, :].T.dot(lam[:, ichannel, newaxis] * h[ichannel, :])
                # TODO: inactive neurons never fire across all trials which may cause zero Hessian
                if adjhess:
                        b[:, ichannel] += solve(neghess_b + diag(sqrt(eps + db_acc[:, ichannel])), grad_b, sym_pos=True)
                else:
                    try:
                        b[:, ichannel] += solve(neghess_b, grad_b, sym_pos=True)
                    except LinAlgError:
                        b[:, ichannel] += solve(neghess_b + eps * identity(b.shape[0]), grad_b, sym_pos=True)
                    else:
                        pass
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
            print('Undefined channel')
    obj['noise'] = var(y - eta, axis=0, ddof=0)

    # constrain loading
    a /= norm(a, ord=inf, axis=1)[..., newaxis]


def fillargs(**kwargs):
    """Fill default values of controlling arguments if missing
    Args:
        **kwargs: optional arguments controlling inference

    Returns:
        valid arguments
    """
    kwargs['verbose'] = kwargs.get('verbose', False)
    kwargs['niter'] = kwargs.get('niter', 50)
    kwargs['infer'] = kwargs.get('infer', 'both')
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
    kwargs['adjhess'] = kwargs.get('adjhess', False)
    return kwargs


def infer(obj, fstat=None, **kwargs):
    """Main inference procedure
    Args:
        obj: inference object
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

    # fully optimize latent at initialization
    # old_elbo = NINF
    # for _ in range(50):
    #     inferpost(obj, **kwargs, adjhess=False, normalize=False)
    #     new_elbo, _ = elbo(obj)
    #     if new_elbo <= old_elbo:
    #         copyto(obj['mu'], good_mu)
    #         copyto(obj['w'], good_w)
    #         copyto(obj['v'], good_v)
    #         break
    #     old_elbo = new_elbo
    #     copyto(good_mu, obj['mu'])
    #     copyto(good_w, obj['w'])
    #     copyto(good_v, obj['v'])

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
    adjhess = False
    converged = False
    stop = False
    infer_tick = timeit.default_timer()

    if kwargs['verbose']:
        print('\nInference starts')

    while not stop and iiter < kwargs['niter']:
        iter_tick = timeit.default_timer()

        # infer posterior
        post_tick = timeit.default_timer()
        if kwargs['infer'] != 'param':
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
        if kwargs['infer'] != 'posterior':
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
        # stop = converged or (decreased and backtrack)

        # keep optimizing loading after latent convergence
        # if stop and kwargs['infer'] == 'both' and kwargs['moreparam']:
        #     if kwargs['verbose']:
        #         print('\nLoading only')
        #     kwargs['infer'] = 'param'
        #     kwargs['param_opt'] = 'GA'
        #     kwargs['tol'] *= 0.1
        #     stop = False

        if decreased:
            if kwargs['verbose']:
                print('\nELBO decreased. Backtracking.')
            backtrack = True
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

        stat[iiter] = fstat(obj) if fstat is not None else {}
        stat[iiter]['Elapsed Post'] = elapsed[iiter, 0]
        stat[iiter]['Elapsed Param'] = elapsed[iiter, 1]
        stat[iiter]['Elapsed Total'] = elapsed[iiter, 2]
        stat[iiter]['ELBO'] = lb[iiter]
        stat[iiter]['LL'] = ll[iiter]
        stat[iiter]['sigma'] = good_sigma
        stat[iiter]['omega'] = good_omega

        if kwargs['verbose']:
            print('\n[{}]'.format(iiter))
            for k, v in stat[iiter].items():
                print('{}: {}'.format(k, v))

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


def infer2(obj, fstat=None, **kwargs):
    """Main inference procedure
    Args:
        obj: inference object
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
    lb[0], ll[0] = elbo(obj)
    # lb[0], ll[0] = NINF, NINF
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
    adjhess = False
    converged = False
    stop = False
    infer_tick = timeit.default_timer()

    if kwargs['verbose']:
        print('\nInference starts')

    while not stop and iiter < kwargs['niter']:
        iter_tick = timeit.default_timer()

        # infer posterior
        post_tick = timeit.default_timer()
        if kwargs['infer'] != 'param':
            inferpost(obj, **kwargs, adjhess=False)
        elbo(obj)
        post_tock = timeit.default_timer()
        elapsed[iiter, 0] = post_tock - post_tick

        lb_post, ll_post = elbo(obj)
        if lb_post < lb[iiter - 1]:
            if kwargs['verbose']:
                print('\nELBO decreased by posterior. Backtracking.')
            copyto(obj['mu'], good_mu)
            copyto(obj['w'], good_w)
            copyto(obj['v'], good_v)
            lb_post = lb[iiter - 1]
            ll_post = ll[iiter - 1]

        # Calculate angle between latent subspace if true latent is given.
        if x is not None:
            for itrial in range(x.shape[0]):
                rotated[itrial, :] = rotate(add_constant(obj['mu'][itrial, :]), x[itrial, :])
            latent_angle[iiter] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))

        # infer parameter
        param_tick = timeit.default_timer()
        if kwargs['infer'] != 'posterior':
            inferparam(obj, **kwargs, adjhess=adjhess)
        param_tock = timeit.default_timer()
        elapsed[iiter, 1] = param_tock - param_tick

        lb_param, ll_param = elbo(obj)
        if lb_param < lb_post:
            if kwargs['verbose']:
                print('\nELBO decreased by parameter. Backtracking.')
            copyto(obj['a'], good_a)
            copyto(obj['b'], good_b)
            copyto(obj['noise'], good_noise)
            lb_param = lb_post
            ll_param = ll_post

        # Calculate angle between loading subspace if true loading is given.
        if alpha is not None:
            loading_angle[iiter] = subspace(alpha.T, obj['a'].T)

        lb[iiter], ll[iiter] = lb_param, ll_param

        converged = abs(lb[iiter] - lb[iiter - 1]) < kwargs['tol'] * abs(lb[iiter - 1])
        stop = converged
        # stop = converged or (decreased and backtrack)

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

        stat[iiter] = fstat(obj) if fstat is not None else {}
        stat[iiter]['Elapsed Post'] = elapsed[iiter, 0]
        stat[iiter]['Elapsed Param'] = elapsed[iiter, 1]
        stat[iiter]['Elapsed Total'] = elapsed[iiter, 2]
        stat[iiter]['ELBO'] = lb[iiter]
        stat[iiter]['LL'] = ll[iiter]
        stat[iiter]['sigma'] = good_sigma
        stat[iiter]['omega'] = good_omega

        if kwargs['verbose']:
            print('\n[{}]'.format(iiter))
            for k, v in stat[iiter].items():
                print('{}: {}'.format(k, v))

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


def fit(y, channel, sigma, omega, x=None, a0=None, mu0=None, alpha=None, beta=None, lag=0, rank=500, **kwargs):
    """Inference API
    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior timescale
        x:       optional true latent
        alpha:   optional true loading
        beta:    optional true autoregression coefficients and bias
        lag:     autoregressive lag
        rank:    prior covariance rank
        **kwargs: optional arguments controlling inference

    Returns:
        inference object
    """
    assert sigma.shape == omega.shape
    kwargs = fillargs(**kwargs)

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., None]
    if y.ndim < 3:
        y = y[None, ...]
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
    a = fa.components_ * norm(mu, ord=inf, axis=0, keepdims=True).T
    # constrain loading and center latent
    # mu /= norm(a, ord=inf, axis=1)
    mu -= mu.mean(axis=0)
    a /= norm(a, ord=inf, axis=1)[..., newaxis]
    # mu /= norm(mu, ord=inf, axis=0)
    mu = mu.reshape((ntrial, ntime, nlatent))

    if a0 is not None:
        a = a0
    if mu0 is not None:
        mu = mu0

    L = empty((ntrial, nlatent, ntime, rank))

    # initialize parameters
    b = empty((1 + lag, nchannel), dtype=float)
    for ichannel in range(nchannel):
        b[:, ichannel] = lstsq(h.reshape((nchannel, -1, 1 + lag))[ichannel, :], y.reshape((-1, nchannel))[:, ichannel])[
            0]

    noise = var(y.reshape((-1, nchannel)), axis=0, ddof=0)

    obj = {'y': y, 'channel': channel, 'h': h,
           'sigma': sigma, 'omega': omega, 'chol': chol, 'sigma0': sigma, 'omega0': omega,
           'mu': mu, 'w': zeros((ntrial, ntime, nlatent)), 'v': zeros((ntrial, ntime, nlatent)), 'L': L, 'x': x,
           'a': a, 'b': b, 'noise': noise, 'alpha': alpha, 'beta': beta}

    kwargs['dmu_acc'] = zeros_like(mu)
    kwargs['da_acc'] = zeros_like(a)
    kwargs['db_acc'] = zeros_like(b)

    inference = postprocess(infer(obj, **kwargs))
    return inference


def fit2(y, channel, sigma, omega, x=None, a0=None, mu0=None, alpha=None, beta=None, lag=0, rank=500, **kwargs):
    """Inference API
    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior timescale
        x:       optional true latent
        alpha:   optional true loading
        beta:    optional true autoregression coefficients and bias
        lag:     autoregressive lag
        rank:    prior covariance rank
        **kwargs: optional arguments controlling inference

    Returns:
        inference object
    """
    assert sigma.shape == omega.shape
    kwargs = fillargs(**kwargs)

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., None]
    if y.ndim < 3:
        y = y[None, ...]
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
    mu -= mu.mean(axis=0)
    a = fa.components_ * norm(mu, ord=inf, axis=0, keepdims=True).T
    mu /= norm(mu, ord=inf, axis=0)
    mu = mu.reshape((ntrial, ntime, nlatent))

    if a0 is not None:
        a = a0
    if mu0 is not None:
        mu = mu0

    L = empty((ntrial, nlatent, ntime, rank))

    # initialize parameters
    b = empty((1 + lag, nchannel), dtype=float)
    for ichannel in range(nchannel):
        b[:, ichannel] = lstsq(h.reshape((nchannel, -1, 1 + lag))[ichannel, :], y.reshape((-1, nchannel))[:, ichannel])[
            0]

    noise = var(y.reshape((-1, nchannel)), axis=0, ddof=0)

    obj = {'y': y, 'channel': channel, 'h': h,
           'sigma': sigma, 'omega': omega, 'chol': chol, 'sigma0': sigma, 'omega0': omega,
           'mu': mu, 'w': zeros((ntrial, ntime, nlatent)), 'v': zeros((ntrial, ntime, nlatent)), 'L': L, 'x': x,
           'a': a, 'b': b, 'noise': noise, 'alpha': alpha, 'beta': beta}

    kwargs['dmu_acc'] = zeros_like(mu)
    kwargs['da_acc'] = zeros_like(a)
    kwargs['db_acc'] = zeros_like(b)

    inference = postprocess(infer2(obj, **kwargs))
    return inference


def fitpost(y, channel, sigma, omega, a, b, mu0=None, x=None, alpha=None, beta=None, lag=0, rank=500, **kwargs):
    """Inference API
    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior timescale
        mu:       optional true latent
        alpha:   optional true loading
        beta:    optional true autoregression coefficients and bias
        lag:     autoregressive lag
        rank:    prior covariance rank
        **kwargs: optional arguments controlling inference

    Returns:
        inference object
    """
    assert sigma.shape == omega.shape
    kwargs = fillargs(**kwargs)

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., None]
    if y.ndim < 3:
        y = y[None, ...]
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
    mu = lstsq(a.T, y.reshape((-1, nchannel)).T)[0].T.reshape((ntrial, ntime, -1)) if mu0 is None else mu0.copy()
    L = empty((ntrial, nlatent, ntime, rank))

    noise = var(y.reshape((-1, nchannel)), axis=0, ddof=0)

    obj = {'y': y, 'channel': channel, 'h': h,
           'sigma': sigma, 'omega': omega, 'chol': chol, 'sigma0': sigma, 'omega0': omega,
           'mu': mu, 'w': zeros((ntrial, ntime, nlatent)), 'v': zeros((ntrial, ntime, nlatent)), 'L': L,
           'a': a, 'b': b, 'noise': noise, 'x': x, 'alpha': alpha, 'beta': beta}

    kwargs['dmu_acc'] = zeros_like(mu)
    kwargs['da_acc'] = zeros_like(a)
    kwargs['db_acc'] = zeros_like(b)
    kwargs['infer'] = 'posterior'

    inference = postprocess(infer(obj, **kwargs))
    return inference


def fitparam(y, channel, sigma, omega, mu, x=None, alpha=None, beta=None, lag=0, rank=500, **kwargs):
    """Inference API
    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior timescale
        mu:       optional true latent
        alpha:   optional true loading
        beta:    optional true autoregression coefficients and bias
        lag:     autoregressive lag
        rank:    prior covariance rank
        **kwargs: optional arguments controlling inference

    Returns:
        inference object
    """
    assert sigma.shape == omega.shape
    kwargs = fillargs(**kwargs)

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., None]
    if y.ndim < 3:
        y = y[None, ...]
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
    a = lstsq(mu.reshape((-1, nlatent)), y.reshape((-1, nchannel)))[0]
    L = empty((ntrial, nlatent, ntime, rank))

    # initialize parameters
    b = empty((1 + lag, nchannel), dtype=float)
    for ichannel in range(nchannel):
        b[:, ichannel] = lstsq(h.reshape((nchannel, -1, 1 + lag))[ichannel, :], y.reshape((-1, nchannel))[:, ichannel])[
            0]

    noise = var(y.reshape((-1, nchannel)), axis=0, ddof=0)

    obj = {'y': y, 'channel': channel, 'h': h,
           'sigma': sigma, 'omega': omega, 'chol': chol, 'sigma0': sigma, 'omega0': omega,
           'mu': mu, 'w': zeros((ntrial, ntime, nlatent)), 'v': zeros((ntrial, ntime, nlatent)), 'L': L,
           'a': a, 'b': b, 'noise': noise, 'x': x, 'alpha': alpha, 'beta': beta}

    kwargs['dmu_acc'] = zeros_like(mu)
    kwargs['da_acc'] = zeros_like(a)
    kwargs['db_acc'] = zeros_like(b)
    kwargs['infer'] = 'param'

    inference = postprocess(infer(obj, **kwargs))
    return inference


def seqfit(y, channel, sigma, omega, lag=0, rank=500, **kwargs):
    """Sequential inference API
    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior timescale
        lag:     autoregressive lag
        rank:    prior covariance rank
        **kwargs: optional arguments controlling inference

    Returns:
        list of inference objects
    """
    assert sigma.shape == omega.shape
    kwargs = fillargs(**kwargs)

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., None]
    if y.ndim < 3:
        y = y[None, ...]
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
    mu -= mu.mean(axis=0)
    a = fa.components_ * norm(mu, ord=inf, axis=0, keepdims=True).T
    mu /= norm(mu, ord=inf, axis=0)
    mu = mu.reshape((ntrial, ntime, nlatent))
    L = empty((ntrial, nlatent, ntime, rank))
    w = zeros((ntrial, ntime, nlatent))
    v = zeros((ntrial, ntime, nlatent))

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
        # copyto(mu[:, :, ilatent], mu[:, :, 0])
        # copyto(w[:, :, ilatent], w[:, :, 0])
        # copyto(v[:, :, ilatent], v[:, :, 0])
        # copyto(chol[ilatent, :], chol[0, :])
        # copyto(sigma[ilatent], sigma[0])
        # copyto(omega[ilatent], omega[0])
        obj = {'y': y, 'channel': channel, 'h': h,
               'sigma': sigma[:ilatent + 1].copy(), 'omega': omega[:ilatent + 1].copy(),
               'chol': chol[:ilatent + 1, :].copy(),
               'mu': mu[:, :, :ilatent + 1], 'w': w[:, :, :ilatent + 1], 'v': v[:, :, :ilatent + 1],
               'a': a[:ilatent + 1, :], 'b': b, 'noise': noise}

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
    kwargs = fillargs(**kwargs)
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
            fa = FactorAnalysis(n_components=nlatent, svd_method='lapack')
            mu = fa.fit_transform(obj['y'].reshape((-1, nchannel - 1)))
            mu -= mu.mean(axis=0)
            mu /= norm(mu, ord=inf, axis=0)
            mu = mu.reshape((ntrial, ntime, nlatent))
        else:
            mu = trial['mu0']
        obj['mu'] = mu
        obj['w'] = zeros((ntrial, ntime, nlatent))
        obj['v'] = zeros((ntrial, ntime, nlatent))
        obj['sigma'] = model['sigma'].copy()
        obj['omega'] = model['omega'].copy()
        obj['chol'] = model['chol'].copy()
        obj['L'] = model['L'].copy()

        # set parameters
        obj['a'] = model['a'][:, included]
        obj['b'] = model['b'][:, included]
        obj['noise'] = model['noise'][included]
        kwargs['hyper'] = False
        kwargs['infer'] = 'posterior'

        obj = infer(obj, **kwargs)

        if channel[ichannel] == 'spike':
            yhat[:, :, ichannel] = exp(
                    obj['mu'].reshape((-1, nlatent)).dot(a[:, ichannel]) + htest.reshape((ntime, -1)).dot(
                            b[:, ichannel]))
        else:
            yhat[:, :, ichannel] = obj['mu'].reshape((-1, nlatent)).dot(a[:, ichannel]) + htest.reshape(
                    (ntime, -1)).dot(b[:, ichannel])

    return trial


def cv(y, channel, sigma, omega, a0=None, mu0=None, lag=0, rank=500, **kwargs):
    """Cross-validation
    Do leave-one-out prediction to all trials. Use one trial as test and the rest as training each time.

    Args:
        y:       observation matrix
        channel: channel type indicator (spike/lfp)
        sigma:   initial prior variance
        omega:   initial prior time scale
        lag:     autoregressive lag
        rank:    prior covariance rank
        **kwargs: optional arguments controlling inference

    Returns:
        prediction of all neurons
    """
    kwargs = fillargs(**kwargs)
    assert sigma.shape == omega.shape

    y = asarray(y)
    if y.ndim < 2:
        y = y[..., None]
    if y.ndim < 3:
        y = y[None, ...]
    channel = asarray(channel)
    ntrial, ntime, nchannel = y.shape

    h = empty((nchannel, ntrial, ntime, 1 + lag), dtype=float)
    for ichannel in range(nchannel):
        for itrial in range(ntrial):
            h[ichannel, itrial, :] = add_constant(lagmat(y[itrial, :, ichannel], lag=lag))

    yhat = empty_like(y, dtype=float)
    # do leave-one-out trial by trial
    for itrial in range(ntrial):
        test_trial = {'y': y[itrial, :][None, ...], 'h': h[:, itrial, :, :][:, None, :, :],
                      'yhat': yhat[itrial, :][None, ...], 'mu0': mu0[itrial, :][None, ...] if mu0 is not None else None}
        itrain = arange(ntrial) != itrial
        model = fit(y[itrain, :], channel, sigma, omega, x=None, a0=a0, mu0=mu0[itrain, :], alpha=None, beta=None, lag=lag, rank=rank, **kwargs)
        kwargs['verbose'] = False
        kwargs['hyper'] = False
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
            A = eyer - GtWG + GtWG.dot(solve(eyer + GtWG, GtWG, sym_pos=True))  # A should be PD but numerically not
            eigval, eigvec = eigh(A)
            eigval.clip(0, PINF, out=eigval)  # remove negative eigenvalues
            L[itrial, ilatent, :] = G.dot(eigvec.dot(diag(sqrt(eigval))))  # lower posterior covariance
    obj['L'] = L
    keys = list(obj.keys())
    for key in keys:
        if obj.get(key, None) is None:
            obj.pop(key, None)
    obj.pop('h', None)
    obj.pop('stat', None)
    return obj
