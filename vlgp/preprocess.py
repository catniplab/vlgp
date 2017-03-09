import numpy as np

from vlgp.math import ichol_gauss
from .constant import REQUIRED_FIELDS, TYPE_CODE, DEFAULT_VALUES
from .name import Y_TYPE
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
    ntrial, nbin, y_ndim = model['y'].shape

    # sanity check
    z_ndim = model['dyn_ndim']
    if y_ndim < z_ndim:
        raise ValueError("The number of observation dimensions, {}, is less "
                         "than that of latent dimensions, {}.".format(y_ndim, z_ndim))

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

    rank = model['rank']

    model['chol'] = np.array([ichol_gauss(nbin, w, rank) * s for s, w in
                              zip(model['sigma'], model['omega'])])

    # cut trials
    from .util import cut_trials
    if model.get('segment') is None:
        model['segment'] = cut_trials(nbin, ntrial, seg_len=model['seg_len'])
