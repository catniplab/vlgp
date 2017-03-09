import numpy as np
from numpy import empty, var, zeros_like
from scipy.linalg import lstsq
from sklearn.decomposition import FactorAnalysis

from .constant import SPIKE, UNUSED
from .core import update_w, update_v
from .math import ichol_gauss
from .util import add_constant, lagmat
from .constant import REQUIRED_FIELDS, TYPE_CODE, DEFAULT_VALUES
from .name import Y_TYPE


def build_model(**kwargs):
    model = kwargs
    check_model(model)
    return model


def check_model(model):
    missing_fields = [field for field in REQUIRED_FIELDS if
                      model.get(field) is None]
    if missing_fields:
        raise ValueError('{} missed'.format(missing_fields))

    # default values
    for k, v in DEFAULT_VALUES.items():
        # If key is in the dictionary, return its value.
        # If not, insert key with a value of default and return default.
        model.setdefault(k, v)
        # TODO: remove options, flatten model
        # model['options'].setdefault(k, v)

    # make sure y is numpy array of compatible shape
    y = np.asarray(model['y'])
    y = y.astype(float)
    if y.ndim < 2:
        y = y[..., np.newaxis]
    if y.ndim < 3:
        y = y[np.newaxis, ...]

    # y-dependent arguments
    ntrial, nbin, y_ndim = model['y'].shape

    # sanity check
    if y_ndim < model['dyn_ndim']:
        raise ValueError("The number of observation dimensions, {}, is less "
                         "than that of latent dimensions, {}.".format(y_ndim, model['dyn_ndim']))

    types = model.get(Y_TYPE)
    if types is None:
        types = 'spike'

    if types == 'spike' or types == 'lfp':
        types = [types] * y_ndim

    encoded_types = np.empty(y_ndim, dtype=int)
    for i, t in enumerate(types):
        encoded_types[i] = TYPE_CODE[t]

    model[Y_TYPE] = encoded_types

    # parameters
    model.setdefault('a', None)
    model.setdefault('b', None)
    model.setdefault('mu', None)

    # IDs of trials and neurons or lfp channels
    # for identification in case of bad data
    model.setdefault('trial_id', np.arange(ntrial))
    model.setdefault('obs_id', np.arange(y_ndim))

    # length of history filter
    model.setdefault('history', 0)

    # make design matrix of regression
    h = model.get('h')
    if h is None:
        history = model['history']
        h = np.empty((y_ndim, ntrial, nbin, 1 + history), dtype=float)
        for y_dim in range(y_ndim):
            for trial in range(ntrial):
                h[y_dim, trial, :] = add_constant(
                    lagmat(y[trial, :, y_dim], lag=history))
        model['h'] = h

    # GP
    model.setdefault('rank', nbin // 5)

    sigma = np.empty(model['dyn_ndim'])
    sigma[:] = np.asarray(model['sigma'])
    model['sigma'] = sigma

    omega = model.get('omega')
    if omega is None:
        model['omega'] = np.full(model['dyn_ndim'],
                                 fill_value=0.5 / (model['tau'] ** 2))
    else:
        model['omega'] = np.empty(model['dyn_ndim'])
        model['omega'][:] = np.asarray(omega)


def initialize(model):
    y = model['y']
    h = model['h']
    a = model['a']
    b = model['b']
    mu = model['mu']
    sigma = model['sigma']
    omega = model['omega']

    ntrial, nbin, y_ndim = y.shape
    history = model['history']
    z_ndim = model['dyn_ndim']

    y_2d = y.reshape((-1, y_ndim))

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent
    if a is None and mu is None:
        fa = FactorAnalysis(n_components=z_ndim, svd_method='lapack')
        y0 = y[0, :]
        fa.fit(y0)
        a = fa.components_
        mu_2d = fa.transform(y_2d)

        # constrain loading and center latent
        # scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
        # a /= scale
        # mu *= scale.squeeze()  # compensate latent
        # mu -= mu.mean(axis=0)

        mu = mu_2d.reshape((ntrial, nbin, z_ndim))

        # noinspection PyTupleAssignmentBalance
        # U, s, Vh = svd(a, full_matrices=False)
        # mu = np.reshape(mu @ a @ Vh.T, (ntrial, nbin, nlatent))
        # a[:] = Vh
    else:
        if mu is None:
            mu = lstsq(a.T, y_2d.T)[0].T.reshape((ntrial, nbin, z_ndim))
        elif a is None:
            a = lstsq(mu.reshape((-1, z_ndim)), y_2d)[0]

    # initialize regression
    # if b is None:
    #     b = leastsq(h, y)
    spike = model[Y_TYPE] == SPIKE

    if b is None:
        b = empty((1 + history, y_ndim), dtype=float)
        for y_dim in np.arange(y_ndim)[spike]:
            b[:, y_dim] = \
                lstsq(h.reshape((y_ndim, -1, 1 + history))[y_dim, :],
                      y.reshape((-1, y_ndim))[:, y_dim])[0]

    # initialize noises of LFP
    model['noise'] = var(y_2d, axis=0, ddof=0)
    model[Y_TYPE][model['noise'] == 0] = UNUSED
    a[:, model[Y_TYPE] == UNUSED] = 0
    b[:, model[Y_TYPE] == UNUSED] = 0

    ####################
    # initialize prior #
    ####################

    rank = model['rank']

    prior = np.array(
        [ichol_gauss(nbin, omega[z_dim], rank) * sigma[z_dim] for z_dim in
         range(z_ndim)])

    # fill model fields
    model['a'] = a
    model['b'] = b
    model['mu'] = mu
    model['w'] = zeros_like(mu, dtype=float)
    model['v'] = zeros_like(mu, dtype=float)
    model['chol'] = prior

    model['dmu'] = zeros_like(model['mu'])
    model['da'] = zeros_like(model['a'])
    model['db'] = zeros_like(model['b'])

    update_w(model)
    update_v(model)

    # cut trials
    from .util import cut_trials
    model['segment'] = cut_trials(nbin, ntrial, seg_len=model['seg_len'])
