import numpy as np


def choice_prob(x, y, sd=False):
    """
    Computes the choice probability
    """
    from scipy.stats import rankdata

    x = np.asarray(x)
    y = np.asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1 * n2 - u1  # remainder is U for y

    if sd:
        return 1 - u2 / n1 / n2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) / n1 / n2

    return 1 - u2 / n1 / n2


def choice_corr(x, y, fit_intercept=True):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    lm = LinearRegression(fit_intercept=fit_intercept)
    lm.fit(x, y)
    return r2_score(y_true=y, y_pred=lm.predict(x))


def spontaneous(loading, size):
    from scipy.linalg import svd
    _, _, vt = svd(loading, full_matrices=False)
    mu = loading @ vt.T @ np.random.randn(loading.shape[1], size) + 10
    # mu -= np.min(mu)
    return np.random.poisson(mu).T
