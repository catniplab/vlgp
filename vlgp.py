"""
Functions of Inference
"""
import timeit

from numpy import identity, einsum, trace, inner, empty, mean, inf, diag, newaxis, var, asarray, zeros, zeros_like, \
    empty_like, arange, sum
from numpy.core.umath import sqrt, PINF, log
from numpy.linalg import norm, slogdet
from scipy import stats
from scipy.linalg import lstsq, eigh, solve
from sklearn.decomposition.factor_analysis import FactorAnalysis

from hyper import learngp
from mathf import ichol_gauss, subspace, sexp
from util import add_constant, rotate, lagmat


def elbo(obj):
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
        L = obj['L'][itrial, :]

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
            mu[:, ilatent] -= mean(mu[:, ilatent])
            scale = norm(mu[:, ilatent], ord=inf)
            mu[:, ilatent] /= scale

        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        U[:, spike] = lam[:, spike]
        U[:, lfp] = 1 / noise[lfp]
        w[:, :] = U.dot(a.T ** 2)


def inferparam(obj, **kwargs):
    nchannel, ntrial, ntime, lag1 = obj['h'].shape  # neuron, trial, time, lag + 1
    nlatent, _, rank = obj['chol'].shape  # latent, time, rank

    y = obj['y'].reshape((-1, nchannel))  # concatenate trials
    h = obj['h'].reshape((nchannel, -1, lag1))  # concatenate trials
    channel = obj['channel']

    mu = obj['mu'].reshape((-1, nlatent))
    v = obj['v'].reshape((-1, nlatent))

    a = obj['a']
    b = obj['b']
    noise = obj['noise']

    decay = kwargs['decay']
    adjhess = kwargs['adjhess']
    eps = kwargs['eps']
    da_acc = kwargs['da_acc']
    db_acc = kwargs['db_acc']

    for ichannel in range(nchannel):
        eta = mu.dot(a) + einsum('ijk, ki -> ji', h, b)
        lam = sexp(eta + 0.5 * v.dot(a ** 2))
        if channel[ichannel] == 'spike':
            # a
            va = v * a[:, ichannel]  # (ntime, nlatent)
            wv = diag(lam[:, ichannel].dot(v))
            grad_a = mu.T.dot(y[:, ichannel]) - (mu + va).T.dot(lam[:, ichannel])
            da_acc[:, ichannel] = accumulate(da_acc[:, ichannel], grad_a, decay)

            neghess_a = (mu + va).T.dot(lam[:, ichannel, newaxis] * (mu + va)) + wv
            if adjhess:
                delta_a = solve(neghess_a + diag(sqrt(eps + da_acc[:, ichannel])), grad_a, sym_pos=True)
            else:
                delta_a = solve(neghess_a, grad_a, sym_pos=True)
            a[:, ichannel] += delta_a

            # b
            grad_b = h[ichannel, :].T.dot(y[:, ichannel] - lam[:, ichannel])
            db_acc[:, ichannel] = accumulate(db_acc[:, ichannel], grad_b, decay)
            neghess_b = h[ichannel, :].T.dot(lam[:, ichannel, newaxis] * h[ichannel, :])
            if adjhess:
                b[:, ichannel] += solve(neghess_b + diag(sqrt(eps + db_acc[:, ichannel])), grad_b, sym_pos=True)
            else:
                b[:, ichannel] += solve(neghess_b, grad_b, sym_pos=True)
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
    noise[:] = var(y - eta, axis=0, ddof=0)


def fillargs(**kwargs):
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
    return kwargs


def infer(obj, fstat=None, **kwargs):
    """Main inference procedure

    Args:
        obj:
        kwargs:

    Returns:

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
    if alpha is not None:
        loading_angle[0] = subspace(alpha.T, obj['a'].T)
    if x is not None:
        rotated = empty_like(x)
        # rotate trial by trial
        for itrial in range(x.shape[0]):
            rotated[itrial, :] = rotate(add_constant(obj['mu'][itrial, :]), x[itrial, :])
        latent_angle[0] = subspace(rotated.reshape((-1, x.shape[-1])), x.reshape((-1, x.shape[-1])))

    #
    iiter = 1
    adjhess = False
    converged = False
    infer_tick = timeit.default_timer()

    if kwargs['verbose']:
        print('\nInference starts')

    while not converged and iiter < kwargs['niter']:
        iter_tick = timeit.default_timer()

        # infer posterior
        post_tick = timeit.default_timer()
        if kwargs['infer'] != 'param':
            inferpost(obj, **kwargs, adjhess=adjhess)
        post_tock = timeit.default_timer()
        elapsed[iiter, 0] = post_tock - post_tick
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
        if alpha is not None:
            loading_angle[iiter] = subspace(alpha.T, obj['a'].T)

        # if iiter % kwargs['nhyper'] == 0 and (kwargs['learn_sigma'] or kwargs['learn_omega']):
        #     gp = learngp(obj, **kwargs)
        #     obj['sigma'][:] = gp[0]
        #     obj['omega'][:] = gp[1]
        #     # if kwargs['verbose']:
        #     #     print('sigma: {} \nomega: {}'.format(obj['sigma'], obj['omega']))
        #     for ilatent in range(nlatent):
        #         obj['chol'][ilatent, :] = ichol_gauss(ntime, obj['omega'][ilatent], rank) * obj['sigma'][ilatent]

        lb[iiter], ll[iiter] = elbo(obj)
        if lb[iiter] < lb[iiter - 1]:
            # backtracking
            obj['mu'][:] = good_mu
            obj['w'][:] = good_w
            obj['v'][:] = good_v
            obj['a'][:] = good_a
            obj['b'][:] = good_b
            # obj['noise'][:] = good_noise
            # obj['sigma'][:] = good_sigma
            # obj['omega'][:] = good_omega
            for ilatent in range(nlatent):
                obj['chol'][ilatent, :] = ichol_gauss(ntime, obj['omega'][ilatent], rank) * obj['sigma'][ilatent]

            lb[iiter] = lb[iiter - 1]
            if kwargs['verbose']:
                print('\nELBO decreased. Backtracking.')
            if iiter > kwargs['nadjhess'] and not adjhess:
                adjhess = True
                if kwargs['verbose']:
                    print('\nHessian adjustment enabled.')
            else:
                converged = True
        elif abs(lb[iiter] - lb[iiter - 1]) < kwargs['tol'] * abs(lb[iiter - 1]):
            converged = True
        else:
            adjhess = False

        if iiter % kwargs['nhyper'] == 0 and (kwargs['learn_sigma'] or kwargs['learn_omega']):
            gp = learngp(obj, **kwargs)
            obj['sigma'][:] = gp[0]
            obj['omega'][:] = gp[1]
            # if kwargs['verbose']:
            #     print('sigma: {} \nomega: {}'.format(obj['sigma'], obj['omega']))
            for ilatent in range(nlatent):
                obj['chol'][ilatent, :] = ichol_gauss(ntime, obj['omega'][ilatent], rank) * obj['sigma'][ilatent]


        good_mu[:] = obj['mu']
        good_w[:] = obj['w']
        good_v[:] = obj['v']
        good_a[:] = obj['a']
        good_b[:] = obj['b']
        good_noise[:] = obj['noise']
        good_sigma[:] = obj['sigma']
        good_omega[:] = obj['omega']

        iter_tock = timeit.default_timer()
        elapsed[iiter, 2] = iter_tock - iter_tick

        stat[iiter] = fstat(obj) if fstat is not None else {}
        stat[iiter]['Posterior Elapsed'] = elapsed[iiter, 0]
        stat[iiter]['Parameter Elapsed'] = elapsed[iiter, 1]
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


def fit(y, channel, sigma, omega, x=None, alpha=None, beta=None, lag=0, rank=500, **kwargs):
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


def seqfit(y, channel, sigma, omega, lag=0, rank=500, **kwargs):
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
    for i in range(nlatent):
        print('{} latent(s)'.format(i + 1))
        obj = {'y': y, 'channel': channel, 'h': h,
               'sigma': sigma[:i + 1], 'omega': omega[:i + 1], 'chol': chol[:i + 1, :],
               'mu': mu[:, :, :i + 1], 'w': w[:, :, :i + 1], 'v': v[:, :, :i + 1], 'L': L[:, :i + 1, :, :],
               'a': a[:i + 1, :], 'b': b, 'noise': noise}

        kwargs['dmu_acc'] = zeros_like(mu[:, :, :i + 1])
        kwargs['da_acc'] = zeros_like(a[:i + 1, :])
        kwargs['db_acc'] = zeros_like(b)

        objs.append(postprocess(infer(obj, **kwargs)))

    return objs


def leave_one_out(trial, model, **kwargs):
    """Leave-one-out

    Predict each spike train

    Args:
        trial:
        model:
        kwargs:

    Returns:

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
        exceptn = arange(nchannel) != ichannel
        ytrain = y[:, :, exceptn]
        htrain = h[exceptn, :]
        hn = h[ichannel, :]

        obj = {'y': ytrain, 'h': htrain, 'channel': channel[exceptn]}

        # initialize posterior
        fa = FactorAnalysis(n_components=nlatent, svd_method='lapack')
        mu = fa.fit_transform(obj['y'].reshape((-1, nchannel - 1)))
        mu -= mu.mean(axis=0)
        mu /= norm(mu, ord=inf, axis=0)
        mu = mu.reshape((ntrial, ntime, nlatent))
        obj['mu'] = mu
        obj['w'] = zeros((ntrial, ntime, nlatent))
        obj['v'] = zeros((ntrial, ntime, nlatent))
        obj['sigma'] = model['sigma'].copy()
        obj['omega'] = model['omega'].copy()
        obj['chol'] = model['chol'].copy()
        obj['L'] = model['L'].copy()

        # set parameters
        obj['a'] = model['a'][:, exceptn]
        obj['b'] = model['b'][:, exceptn]
        obj['noise'] = model['noise'][exceptn]
        kwargs['hyper'] = False
        kwargs['infer'] = 'posterior'

        obj = infer(obj, **kwargs)

        if channel[ichannel] == 'spike':
            yhat[:, :, ichannel] = sexp(
                    obj['mu'].reshape((-1, nlatent)).dot(a[:, ichannel]) + hn.reshape((ntime, -1)).dot(
                            b[:, ichannel]))
        else:
            yhat[:, :, ichannel] = obj['mu'].reshape((-1, nlatent)).dot(a[:, ichannel]) + hn.reshape(
                    (ntime, -1)).dot(b[:, ichannel])

    return trial


def cv(y, channel, sigma, omega, lag=0, rank=500, **kwargs):
    """Use each trial as testset

    Args:
        y:
        channel:
        sigma:
        omega:
        lag:
        rank:
        niter:
        nadjhess:
        tol:
        hyper:

    Returns:

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
                      'yhat': yhat[itrial, :][None, ...]}
        itrain = arange(ntrial) != itrial
        model = fit(y[itrain, :], channel, sigma, omega, x=None, alpha=None, beta=None, lag=lag, rank=rank, **kwargs)
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
    """
    Remove intermediate and empty variables, and compute decomposition of posterior covariance
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
    return obj
