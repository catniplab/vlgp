from numpy import exp, arcsin
from numpy import sum, dot
from numpy import zeros, ones, diag, meshgrid, arange, eye, asarray
from scipy.linalg import svd, lstsq, orth, norm
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat


def makeregressor(obs, p):
    """Construct full regressive matrix

    Args:
        obs: observations (T, N)
        p: order of auto/cross-regression

    Returns:
        full regressive matrix (T, 1 + p*N)
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
    """
    Construct square exponential covariance matrix
    :param n: size
    :param w: inverse of squared lengthscale
    :param var: variance
    :return: (n, n) covariance matrix
    """

    i, j = meshgrid(arange(n), arange(n))
    return var * exp(- w * (i - j) ** 2)


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
    """Raster plot

    Args:
        y: spike trains (T, N)

    Returns:

    """
    from matplotlib.pyplot import figure, xlim, ylim, vlines, xticks, yticks, gca

    T, N = y.shape
    figure(figsize=(10, 6))
    ylim(0, N)
    xlim(0, T)
    for n in range(N):
        vlines(arange(T)[y[:, n] > 0], n, n + 1, color='black')
    xticks([])
    yticks([])
    gca().invert_yaxis()


def history(obs, lag):
    """Construct autoregressive matrices

    Args:
        obs: observations (T, N)
        lag: order of autoregression

    Returns:
        autoregressive matrices (N, T, 1 + lag)
    """

    T, N = obs.shape
    h = zeros((N, T, 1 + lag), dtype=float)

    for n in range(N):
        h[n, :] = add_constant(lagmat(obs[:, n], maxlag=lag))

    return h


def rotate(obj, ref):
    return obj.dot(lstsq(obj, ref)[0])


def subspace(A, B):
    oA = orth(A)
    oB = orth(B)
    if oA.shape[1] < oB.shape[1]:
        oA, oB = oB.copy(), oA.copy()
    oB -= oA.dot(oA.T.dot(oB))
    return arcsin(min(1, norm(oB, ord=2)))