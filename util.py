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


def varimax(x, gamma=1.0, q=20, tol=1e-5):
    from scipy.linalg import svd
    from numpy import eye, asarray, dot, sum, diag
    p,k = x.shape
    rotation = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        rotated = dot(x, rotation)
        u, s, vh = svd(dot(x.T, asarray(rotated) ** 3 - (gamma / p) * dot(rotated, diag(diag(dot(rotated.T, rotated))))))
        rotation = dot(u, vh)
        d = sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return dot(x, rotation)


def raster(y):
    from matplotlib.pyplot import figure, xlim, ylim, vlines, xticks, yticks, gca
    T, N = y.shape
    figure(figsize=(8, 8))
    ylim(0, N)
    xlim(0, T)
    for n in range(N):
        vlines(np.arange(T)[y[:, n] > 0], n, n + 1, color='black')
    xticks([])
    yticks([])
    gca().invert_yaxis()