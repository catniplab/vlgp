import numpy as np
from numpy import empty, var, zeros_like
from scipy.linalg import lstsq
from sklearn.decomposition import FactorAnalysis

from vlgp.constant import SPIKE, UNUSED
from vlgp.core import update_w, update_v
from vlgp.name import Y_TYPE


def factanal(model):
    """Initialization using factor analysis"""
    y = model['y']
    h = model['h']
    a = model['a']
    b = model['b']
    mu = model['mu']

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

    # fill model fields
    model['a'] = a
    model['b'] = b
    model['mu'] = mu
    model['w'] = zeros_like(mu, dtype=float)
    model['v'] = zeros_like(mu, dtype=float)

    model['dmu'] = zeros_like(model['mu'])
    model['da'] = zeros_like(model['a'])
    model['db'] = zeros_like(model['b'])

    update_w(model)
    update_v(model)
