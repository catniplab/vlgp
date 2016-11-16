import numpy as np
import sklearn.decomposition
from numpy import asarray, newaxis, inf, zeros, var, zeros_like, arange, empty_like
from numpy.core.numeric import empty
from numpy.core.umath import exp
from scipy import stats
from scipy.linalg import norm, lstsq

from .core import check_options, add_constant, lagmat, ichol_gauss, postprocess, vem, check_y_type, initialize
from .api import fit


def seqfit(y, obs_types, dyn_ndim, sigma, omega, history_filter=0, rank=200, copy=False, path=None, **kwargs):
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
    y = asarray(y)
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., newaxis]
    if y.ndim < 3:
        y = y[newaxis, ...]

    obs_types = check_y_type(obs_types)

    ntrial, nbin, obs_ndim = y.shape

    # make design matrix of regression
    h = empty((obs_ndim, ntrial, nbin, 1 + history_filter), dtype=float)
    for obs_dim in range(obs_ndim):
        for trial in range(ntrial):
            h[obs_dim, trial, :] = add_constant(lagmat(y[trial, :, obs_dim], lag=history_filter))

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent

    full_model = dict(y=y,
                 channel=obs_types,
                 dyn_ndim=dyn_ndim,
                 history_filter=history_filter,
                 h=h,
                 sigma=sigma,
                 omega=omega,
                 rank=rank,
                 path=path,
                 options=kwargs)

    initialize(full_model)

    model_seq = []

    for ilatent in range(dyn_ndim):
        print('\n{} latent(s)'.format(ilatent + 1))
        model = dict(y=y,
                     channel=obs_types,
                     dyn_ndim=dyn_ndim,
                     history_filter=history_filter,
                     h=h,
                     sigma=full_model['sigma'][:ilatent + 1],
                     omega=full_model['omega'][:ilatent + 1],
                     chol=full_model['chol'][:ilatent + 1, :],
                     mu=full_model['mu'][:, :, :ilatent + 1],
                     w=full_model['w'][:, :, :ilatent + 1],
                     v=full_model['v'][:, :, :ilatent + 1],
                     a=full_model['a'][:ilatent + 1, :],
                     b=full_model['b'],
                     noise=full_model['noise'],
                     rank=rank,
                     path=path,
                     options=kwargs)
        vem(model)
        model_seq.append(model)

    return model_seq


def leave_one_out(trial, model, **kwargs):
    """Leave-one-out prediction
    Args:
        trial: trial to predict
        model: fitted model
        kwargs: optional arguments controlling inference

    Returns:
        trial with prediction
    """
    kwargs = check_options(**kwargs)
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

        obj = vem(obj)
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
    kwargs = check_options(**kwargs)
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
        model = fit(y[itrain, :], channel, dyn_ndim=dyn_ndim, sigma=sigma, omega=omega, z=None, a0=a0,
                       mu0=mu0[itrain, :] if mu0 is not None else None,
                       alpha=None, beta=None,
                       history_filter=lag, rank=rank, **kwargs)
        kwargs['verbose'] = False
        kwargs['dmu_acc'] = zeros_like(model['mu'])
        kwargs['da_acc'] = zeros_like(model['a'])
        kwargs['db_acc'] = zeros_like(model['b'])
        leave_one_out(test_trial, model, **kwargs)
    ll = stats.poisson.logpmf(y.ravel(), yhat.ravel()).reshape(y.shape)
    prediction = {'y': y, 'yhat': yhat, 'LL': ll}
    return prediction
