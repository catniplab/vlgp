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

def history(y, p, intercept=True):
    T, N = y.shape
    Y = np.ones((T, intercept + p * N), dtype=float)
    for t in range(T):
        if t - p >= 0:
            Y[t, intercept:] = y[t - p:t, :].flatten()  # by row
        else:
            Y[t, intercept + (p - t) * N:] = y[:t, :].flatten()
    return Y
