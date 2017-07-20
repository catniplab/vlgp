import numpy as np

from .constant import REQUIRED_FIELDS, LIK_CODE, DEFAULT_VALUES
from .math import ichol_gauss
from .constant import PRIOR, LIK, Z_DIM
from .util import add_constant, lagmat


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
    ntrial, nbin, y_dim = model['y'].shape

    # sanity check
    z_dim = model[Z_DIM]
    if y_dim < z_dim:
        raise ValueError("The number of observation dimensions, {}, is less "
                         "than that of latent dimensions, {}.".format(y_dim,
                                                                      z_dim))

    lik = model.get(LIK, 'poisson')

    if lik == 'poisson' or lik == 'gaussian':
        lik = [lik] * y_dim

    if model['verbose']:
        print('Likelihood', lik)

    encoded_lik = np.empty(y_dim, dtype=int)
    for i, t in enumerate(lik):
        encoded_lik[i] = LIK_CODE[t]

    model[LIK] = encoded_lik

    # parameters
    model.setdefault('a', None)
    model.setdefault('b', None)
    model.setdefault('mu', None)

    # IDs of trials and neurons or lfp channels
    # for identification in case of bad data
    model.setdefault('trial_id', np.arange(ntrial))
    model.setdefault('obs_id', np.arange(y_dim))

    # length of history filter
    model.setdefault('history', 0)

    # make design matrix of regression
    x = model.get('x')
    if x is None:
        history = model['history']
        x_dim = 1 + history
        x = np.empty((ntrial, nbin, x_dim, y_dim), dtype=float)
        for n in range(y_dim):
            for trl in range(ntrial):
                x[trl, :, :, n] = add_constant(lagmat(y[trl, :, n], lag=history))
        model['x'] = x

    # GP
    model.setdefault('rank', nbin // 5)

    sigma = np.empty(model[Z_DIM])
    sigma[:] = np.asarray(model['sigma'])
    model['sigma'] = sigma

    omega = model.get('omega')
    if omega is None:
        model['omega'] = np.full(model[Z_DIM],
                                 fill_value=0.5 / (model['tau'] ** 2))
    else:
        model['omega'] = np.empty(model[Z_DIM])
        model['omega'][:] = np.asarray(omega)

    rank = model['rank']

    model[PRIOR] = np.array([ichol_gauss(nbin, w, rank) * s for s, w in
                             zip(model['sigma'], model['omega'])])

    # cut trials
    from .util import cut_trials
    if model['seg_len'] > nbin:
        model['seg_len'] = nbin
    if model.get('segment') is None:
        model['segment'] = cut_trials(nbin, ntrial, seg_len=model['seg_len'])
