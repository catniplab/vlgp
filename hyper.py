from numpy import identity, inner, meshgrid, arange, outer, zeros_like, trace, empty_like, empty, sum
from numpy.core.umath import exp, log
from numpy.linalg import slogdet
from scipy.linalg import lstsq, solve
from scipy.optimize import minimize, minimize_scalar
from mathf import ichol_gauss
from util import sqexpcov


def kl(theta, sigma, mu, L):
    """Evidence lower bound for hyperparameter
    Args:
        theta:
        model:

    Returns:

    """
    omega = exp(theta)
    ntrial, ntime = mu.shape
    rank = L.shape[-1]
    # K = sqexpcov(ntime, omega[ilatent], sigma[ilatent] ** 2)
    # S = L[ilatent, :].dot(L[ilatent, :].T)
    # K_mldiv_S = lstsq(K, S)[0]
    G = ichol_gauss(ntime, omega, rank) * sigma
    f = 0.0
    for itrial in range(ntrial):
        G_mldiv_mu = lstsq(G, mu[itrial, :])[0]
        G_mldiv_L = lstsq(G, L[itrial, :])[0]
        K_inv_S = G_mldiv_L.T.dot(G_mldiv_L)
        f += 0.5 * (inner(G_mldiv_mu, G_mldiv_mu) + trace(K_inv_S) - slogdet(K_inv_S)[1])
    return f


def klprime(theta, sigma, mu, L):
    """Gradient of ELBO wrt hyperparameter

    Args:
        omega:
        model:

    Returns:

    """
    omega = exp(theta)
    ntrial, ntime = mu.shape
    rank = L.shape[-1]
    K = sqexpcov(ntime, omega, sigma ** 2) + 1e-6 * identity(ntime)
    dK = Kprime(sigma, theta, ntime)
    KinvdK = solve(K, dK, sym_pos=True)
    # logdetK = slogdet(K)[1]
    df = 0.0
    for itrial in range(ntrial):
        S = L[itrial, :].dot(L[itrial, :].T)
        KinvS = solve(K, S, sym_pos=True)
        df += 0.5 * trace(
            (-solve(K, outer(mu[itrial, :], mu[itrial, :]), sym_pos=True) - KinvS + identity(ntime)).dot(KinvdK))
    return df


def Kprime(sigma, theta, n):
    r, c = meshgrid(arange(n), arange(n))
    difsq = (r - c) ** 2
    return -sigma ** 2 * exp(-exp(theta) * difsq) * exp(theta) * difsq


def learn_hyper(model):
    theta = empty_like(model['omega'])
    nlatent = theta.shape[0]
    for ilatent in range(nlatent):
        theta[ilatent] = minimize(kl, x0=theta[ilatent], method='CG', args=(
        model['sigma'][ilatent], model['mu'][:, :, ilatent], model['L'][:, ilatent, :])).x
    return exp(theta)
