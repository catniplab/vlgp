import numpy as np
from numpy import empty, var, zeros_like
from scipy.linalg import lstsq
from sklearn.decomposition import FactorAnalysis

from .constant import POISSON, NA
from .core import update_w, update_v
from .name import LIK, Z_DIM


def factanal(model):
    """Initialization using factor analysis"""

    y = model['y']
    x = model['x']
    a = model['a']
    b = model['b']
    mu = model['mu']

    ntrial, nbin, y_dim = y.shape
    x_dim = x.shape[-1]
    z_dim = model[Z_DIM]

    x_2d = x.reshape((y_dim, -1, x_dim))
    y_2d = y.reshape((-1, y_dim))

    # Initialize posterior and loading
    # Use factor analysis if both missing initial values
    # Use least squares if missing one of loading and latent
    if a is None and mu is None:
        fa = FactorAnalysis(n_components=z_dim, svd_method='lapack')
        y0 = y[0, :]
        fa.fit(y0)
        a = fa.components_
        mu_2d = fa.transform(y_2d)

        # constrain loading and center latent
        # scale = norm(a, ord=inf, axis=1, keepdims=True) + eps
        # a /= scale
        # mu *= scale.squeeze()  # compensate latent
        # mu -= mu.mean(axis=0)

        mu = mu_2d.reshape((ntrial, nbin, z_dim))

        # noinspection PyTupleAssignmentBalance
        # U, s, Vh = svd(a, full_matrices=False)
        # mu = np.reshape(mu @ a @ Vh.T, (ntrial, nbin, nlatent))
        # a[:] = Vh
    else:
        if mu is None:
            mu = lstsq(a.T, y_2d.T)[0].T.reshape((ntrial, nbin, z_dim))
        elif a is None:
            a = lstsq(mu.reshape((-1, z_dim)), y_2d)[0]

    # initialize regression
    # if b is None:
    #     b = leastsq(h, y)
    poisson = model[LIK] == POISSON

    if b is None:
        b = empty((x_dim, y_dim), dtype=float)
        for n in np.arange(y_dim)[poisson]:
            b[:, n] = lstsq(x_2d[n, :], y_2d[:, n])[0]

    # initialize noises of GAUSSIAN
    model['noise'] = var(y_2d, axis=0, ddof=0)
    model[LIK][model['noise'] == 0] = NA
    a[:, model[LIK] == NA] = 0
    b[:, model[LIK] == NA] = 0

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
