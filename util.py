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
