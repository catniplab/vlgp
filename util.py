from numpy import exp
from numpy import sum, dot
from numpy import zeros, ones, diag, meshgrid, arange, eye, asarray
from scipy.linalg import svd


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


def likelihood(spike, latent, alpha, beta, intercept=True):
    """Poisson likelihood

    Args:
        spike:
        latent:
        alpha:
        beta:
        intercept:

    Returns:

    """

    T, N = spike.shape
    L, _ = latent.shape
    k, _ = beta.shape
    p = (k - intercept) // N

    regressor = makeregressor(spike, p, intercept)

    lograte = dot(regressor, beta) + dot(latent, alpha)
    return sum(spike * lograte - exp(lograte))


# def cartesian(arrays):
#     """Cartesian product
#
#     Args:
#         arrays:
#
#     Returns:
#
#     """
#     arrays = [np.asarray(x) for x in arrays]
#     shape = (len(x) for x in arrays)
#     dtype = arrays[0].dtype
#
#     ix = np.indices(shape)
#     ix = ix.reshape(len(arrays), -1).T
#
#     out = np.empty_like(ix, dtype=dtype)
#
#     for n, arr in enumerate(arrays):
#         out[:, n] = arrays[n][ix[:, n]]
#
#     return out


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


def selfhistory(obs, p, y0=None):
    """Construct autoregressive matrices

    Args:
        obs: observations (T, N)
        p: order of autoregression
        y0: prehistory (N,), 0 if None

    Returns:
        autoregressive matrices (N, T, 1 + p)
    """

    T, N = obs.shape
    h = zeros((N, T, 1 + p), dtype=float)
    if y0 is None:
        y0 = zeros(N, dtype=float)

    for n in range(N):
        for t in range(T):
            h[n, t, 0] = 1
            if t - p < 0:
                h[n, t, 1:p - t + 1] = y0[n]
                h[n, t, p - t + 1:] = obs[:t, n]
            else:
                h[n, t, 1:] = obs[t - p:t, n]

    return h
