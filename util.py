import numpy as np


def makeregressor(spike, p, intercept=True):
    """
    Construct spike regressor.
    :param spike: (T, N) spike trains
    :param p: order of regression
    :param intercept: indicator if intercept term included
    :return: (T, intercept + p * N) array
    """
    T, N = spike.shape
    regressor = np.ones((T, intercept + p * N), dtype=float)
    for t in range(T):
        if t - p >= 0:
            regressor[t, intercept:] = spike[t - p:t, :].flatten()  # by row
        else:
            regressor[t, intercept + (p - t) * N:] = spike[:t, :].flatten()
    return regressor


def sqexpcov(n, w, var=1.0):
    """
    Construct square exponential covariance matrix
    :param n: size
    :param w: inverse of squared lengthscale
    :param var: variance
    :return: (n, n) covariance matrix
    """
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    return var * np.exp(-w * (i - j) ** 2)


def likelihood(spike, latent, alpha, beta, intercept=True):
    T, N = spike.shape
    L, _ = latent.shape
    k, _ = beta.shape
    p = (k - intercept) // N

    regressor = makeregressor(spike, p, intercept)

    lograte = np.dot(regressor, beta) + np.dot(latent, alpha)
    return np.sum(spike * lograte - np.exp(lograte))


def cartesian(arrays):
    """Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out
