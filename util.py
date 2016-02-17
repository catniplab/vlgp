"""
Tool functions
"""
from numpy import exp, column_stack, roll, pi, sum, dot
from numpy import zeros, ones, diag, arange, eye, asarray, atleast_3d, rollaxis
from scipy.linalg import svd, lstsq, toeplitz


def makeregressor(obs, p):
    """Construct full regressive matrix

    Args:
        obs: observations (T, N)
        p: order of auto/cross-regression

    Returns:
        full design matrix (T, 1 + p*N)
    """

    T, N = obs.shape
    regressor = ones((T, 1 + p * N), dtype=float)
    for t in range(T):
        if t - p >= 0:
            regressor[t, 1:] = obs[t - p:t, :].flatten()  # by row
        else:
            regressor[t, 1 + (p - t) * N:] = obs[:t, :].flatten()
    return regressor


def sqexpcov(n, w, var=1.0):
    """Construct square exponential covariance matrix

    Args:
        n: size of the matrix
        w: scale
        var: variance

    Returns:
        covariance
    """

    # i, j = meshgrid(arange(n), arange(n))
    # return var * exp(- w * (i - j) ** 2)
    return var * exp(-w * toeplitz(arange(n)))


def varimax(x, gamma=1.0, q=20, tol=1e-5):
    """Varimax rotation

    Args:
        x: original matrix
        gamma:
        q:
        tol:

    Returns:
        rotated matrix
    """

    p, k = x.shape
    rotation = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        rotated = dot(x, rotation)
        u, s, vh = svd(
            dot(x.T, asarray(rotated) ** 3 - (gamma / p) * dot(rotated, diag(diag(dot(rotated.T, rotated))))))
        rotation = dot(u, vh)
        d = sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return dot(x, rotation)


def history(obs, lag):
    """Construct autoregressive matrices

    Args:
        obs: observations (ntime, nchannel)
        lag: order of autoregression

    Returns:
        autoregression matrices (nchannel, ntime, 1 + lag)
    """

    ntime, nchannel = obs.shape
    h = zeros((nchannel, ntime, 1 + lag), dtype=float)

    for n in range(nchannel):
        h[n, :] = add_constant(lagmat(obs[:, n], lag=lag))

    return h


def regmat(y, lag=0):
    """Autoregression matrices

    Args:
        y: observation
        lag: lag

    Returns:
        autoregression matrices (nchannel, ntrial, ntime, 1 + lag)
    """

    y = asarray(y)
    if y.ndim < 3:
        y = atleast_3d(y)
        y = rollaxis(y, axis=-1)
    ntrial, ntime, nchannel = y.shape
    h = zeros((nchannel, ntrial, ntime, 1 + lag))
    for n in range(nchannel):
        for m in range(ntrial):
            h[n, m, :] = add_constant(lagmat(y[m, :, n], lag=lag))
    return h


def rotate(obj, ref):
    """Rotation

    Args:
        obj:
        ref:

    Returns:

    """
    return obj.dot(lstsq(obj, ref)[0])


def add_constant(x):
    """Add an all-one column to matrix

    Args:
        x: matrix

    Returns:

    """
    x = asarray(x)
    x = column_stack((x, ones((x.shape[0], 1))))
    return roll(x, 1, 1)


def lagmat(x, lag):
    """Make autoregression matrix

    Args:
        x: vector
        lag:

    Returns:

    """
    x = asarray(x)
    if x.ndim < 2:
        x = x[..., None]
    nrow, ncol = x.shape
    if lag >= nrow:
        raise ValueError("lag should be < nrow")
    mat = zeros((nrow + lag, ncol * (lag + 1)))
    for k in range(0, int(lag + 1)):
        mat[lag - k:nrow + lag - k, ncol * (lag - k):ncol * (lag - k + 1)] = x
    startrow = 0
    stoprow = nrow + lag - k

    return mat[startrow:stoprow, ncol:]


def align(x):
    x = asarray(x)
    ax = x.copy()
    if ax.ndim < 3:
        return ax
    else:
        ntrial = ax.shape[0]
        for i in range(1, ntrial):
            diff = ax[i - 1, -1, :] - ax[i, 0, :]
            ax[i, :, :] += diff
    return ax
