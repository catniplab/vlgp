import numpy as np
import sklearn.decomposition
from numpy import asarray, newaxis, inf, zeros, var, zeros_like, arange, empty_like
from numpy.core.numeric import empty
from numpy.core.umath import exp
from scipy import stats
from scipy.linalg import norm, lstsq

from vlgp import fill_options, add_constant, lagmat, ichol_gauss, postprocess, infer, fit


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
    kwargs = fill_options(**kwargs)

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
    kwargs = fill_options(**kwargs)
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
        kwargs['learn_hyper'] = False

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
    kwargs = fill_options(**kwargs)
    assert sigma.shape == omega.shape
    dyn_ndim = sigma.shape[0]

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
        model, _ = fit(y[itrain, :], channel, dyn_ndim=dyn_ndim, sigma=sigma, omega=omega, x=None, a0=a0,
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