from numpy import identity, inner, meshgrid, arange, outer, zeros_like, trace
from numpy.core.umath import exp, log
from numpy.linalg import slogdet
from scipy.linalg import lstsq
from scipy.optimize import minimize

from util import sqexpcov


def hyperelbo(theta, sigma, mu, L):
    """Evidence lower bound for hyperparameter
    Args:
        theta:
        model:

    Returns:

    """
    omega = exp(theta)
    ntime, nlatent = mu.shape
    lb = 0.0
    for ilatent in range(nlatent):
        K = sqexpcov(ntime, omega[ilatent], sigma[ilatent] ** 2)
        S = L[ilatent, :].dot(L[ilatent, :].T)
        K_mldiv_S = lstsq(K, S)[0]
        lb += -0.5 * (inner(mu[:, ilatent], lstsq(K, mu[:, ilatent])[0]) + K_mldiv_S.trace() - slogdet(K_mldiv_S)[1])

    return lb


def hypergrad(theta, sigma, mu, L):
    """Gradient of ELBO wrt hyperparameter

    Args:
        omega:
        model:

    Returns:

    """
    omega = exp(theta)
    ntime, nlatent = mu.shape
    grad = zeros_like(theta)
    for ilatent in range(nlatent):
        K = sqexpcov(ntime, omega[ilatent], sigma[ilatent] ** 2)
        S = L[ilatent, :].dot(L[ilatent, :].T)
        dK = gradK(sigma[ilatent], theta[ilatent], ntime)
        K_mldiv_dK = lstsq(K, dK)[0]
        K_mldiv_S = lstsq(K, S)[0]
        grad[ilatent] = trace(0.5 * (lstsq(K, outer(mu[:, ilatent], mu[:, ilatent]))[0] + K_mldiv_S - identity(ntime)).dot(K_mldiv_dK))

    return grad


def gradK(sigma, theta, n):
    r, c = meshgrid(arange(n), arange(n))
    diff2 = (r - c) ** 2
    return -sigma ** 2 * exp(-exp(theta) * diff2) * exp(theta) * diff2


def learn_hyper(model):
    return minimize(hyperelbo, x0=log(model['omega']), args=(model['sigma'], model['mu'][0, :], model['L'][0, :]),
                    jac=hypergrad, method='CG')
