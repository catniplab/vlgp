import numpy as np


# def history(y, p):
#     if p == 0:
#         return np.empty(0)
#     T, N = y.shape
#     Y = np.zeros((T, p * N), dtype=float)
#     for t in range(T):
#         if t - p >= 0:
#             Y[t, :] = y[t - p:t, :].flatten()  # by row
#         else:
#             Y[t, (p - t) * N:] = y[:t, :].flatten()
#     return Y

def history(spike, p, intercept=True):
    T, N = spike.shape
    regressor = np.ones((T, intercept + p * N), dtype=float)
    for t in range(T):
        if t - p >= 0:
            regressor[t, intercept:] = spike[t - p:t, :].flatten()  # by row
        else:
            regressor[t, intercept + (p - t) * N:] = spike[:t, :].flatten()
    return regressor


def inchol(x, omega, tol=1e-5):
    n = x.shape[x.ndim - 1]
    diag = np.ones(n, dtype=float)
    pvec = np.arange(n, dtype=int)
    i = 0
    g = np.zeros((n, n), dtype=float)
    while np.sum(diag[i:]) > tol:
        jast = np.argmax(diag[i:]) + i
        pvec[i], pvec[jast] = pvec[jast], pvec[i]
        g[jast, :i + 1], g[i, :i + 1] = g[i, :i + 1].copy(), g[jast, :i + 1].copy()
        g[i, i] = np.sqrt(diag[jast])
        g[i + 1:, i] = (np.exp(- omega * np.square(x[pvec[i + 1:]] - x[pvec[i]]))
                        - np.dot(g[i + 1:, :i], g[i, :i].T)) / g[i, i]
        diag[i + 1:] = 1 - np.sum(np.square(g[i + 1:, :i + 1]), axis=1)

        i += 1
    return g, pvec, i - 1

g, pvec, i = inchol(np.arange(5), 1e-4)