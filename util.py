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


def inchol(cov, tol=1e-5):
    diag = np.ones(cov.shape[0])
    pvec = np.arange(diag.size)
    i = 0
    g = np.zeros_like(cov)
    while np.sum(diag[i:]) > tol:
        if i > 0:
            jast = np.argmax(diag[i:]) + i
            pvec[i], pvec[jast] = pvec[jast], pvec[i]
            g[i, :i + 1], g[jast, :i + 1] = g[jast, :i + 1], g[i, :i + 1]
        else:
            jast = 0

        g[i, i] = np.sqrt(diag[jast])
        if i < diag.size - 1:
            newcol = cov[pvec[i + 1:], pvec[i]]
            if i > 0:
                g[i + 1:, i] = (newcol - np.dot(g[i + 1:, :i], g[i, :i].T)) / g[i, i]
            else:
                g[i + 1:, i] = newcol / g[i, i]

        if i < diag.size - 1:
            diag[i + 1:] = 1 - np.sum(np.square(g[i + 1:, :i]), axis=1)

        i += 1
    return g[:, :i], pvec[:i]
