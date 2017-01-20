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


def haefner_cp(cov, weights, var=None, finite=True):
    """computes choice probabilities CP implied by cov and weights
        % if FINITE='finite' (default) a finite population is assumed
        %
        % if FINITE='infinite' an infinitely large population is assumed where
        % cov and weights contain the discretization of the covariance
        % function and the weights function
        %
        % currently works only for 2D C and 1D weights but can be generalized to
        % incorporate time or, in the case of infinitely large populations,
        % multiple dependencies
    """

    var = np.diag(cov) if finite else var

    xsi = (cov @ weights) / np.sqrt(var * (weights.T @ cov @ weights))
    cp = 0.5 + 2 / np.pi * np.sign(xsi) * np.arctan(1. / np.sqrt(2. / xsi ** 2 - 1))  # equation (1) in paper
    return cp


def haefner_weights(corr, cp, use):
    """
        % function [weights V lambda cp_model] = Weights(corr,cp,use)
        %
        % Weight reconstruction from input corr and CPs for a large
        % neuronal population (limit of infinitely many neurons)
        %
        % INPUTS
        % corr: correlation matrix
        % cp: observed choice probabilities
        % use: set of indices of eigenfunctions that are to be used in the
        %      reconstruction (in descending order, i.e. use=[1 2] implies that
        %      only the eigenfunctions with the largest and second-largest
        %      eigenvalue are to be used
        %
        % OUTPUTS
        % weights: implied weights for a large neuronal population
        % V: set of eigenfunctions used for the weight reconstruction
        % lambda : eigenvalues of the eigenfunctions used for weight reconstruction
        % cp_model: choice probabilities implied by the reconstructed weights

    """
    from scipy.linalg import eigh

    w, v = eigh(corr)
    w = w[::-1]
    v = v[:, ::-1]

    nu = 1. / w[use] * (v[:, use] @ (cp - 0.5))
    weights = v[:, use] @ nu

    return weights, v, w, haefner_cp(corr, weights, np.ones(len(cp)), finite=False)
