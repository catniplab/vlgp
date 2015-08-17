import numpy as np


def history(y, p):
    T, N = y.shape
    Y = np.zeros((T, p * N), dtype=float)
    for t in range(T):
        if t - p >= 0:
            Y[t, :] = y[t - p:t, :].flatten()  # vectorized by row
        else:
            Y[t, (p - t) * N:] = y[:t, :].flatten()
    return Y