"""
Cross validation
"""
import numpy as np
import scipy.linalg as lin
from scipy import stats

import vlgp
from vlgp import util


def cv(y, a, b, sigma, omega, nfold):
    ntrial, nbin, nneuron = y.shape  # number of trials, time bins and neurons
    nlatent = len(sigma)

    lag = b.shape[0] - 1

    h = np.empty((nneuron, ntrial, nbin, 1 + lag), dtype=float)
    for neuron in range(nneuron):
        for trial in range(ntrial):
            h[neuron, trial, :] = util.add_constant(util.lagmat(y[trial, :, neuron], lag=lag))

    eta_b = np.einsum('ijk, ki -> ji', h.reshape((nneuron, nbin * ntrial, lag + 1)), b)

    ua, sa, va = lin.svd(a, full_matrices=False)
    orth_a = ua.T.dot(a)

    neuron_for_test = np.array_split(np.arange(nneuron), nfold)
    neuron_for_train = [np.setdiff1d(np.arange(nneuron), each) for each in neuron_for_test]
    ll = []

    for each_train, each_test in zip(neuron_for_train, neuron_for_test):
        ytrain = y[:, :, each_train]
        ytest = y[:, :, each_test]

        model, kwargs = vlgp.fit(ytrain, ['spike'] * ytrain.shape[-1], sigma, omega, rank=100,
                            a=a[:, each_train], b=b[:, each_train], adjhess=True, decay=0,
                            niter=50, tol=1e-5, verbose=True,
                            learn_sigma=False, learn_omega=False, nhyper=5,
                            learn_param=False, learn_posterior=True)

        orth_mu = model['mu'].reshape((-1, nlatent)).dot(ua)
        yhat = [np.exp(orth_mu[:, :i + 1].dot(orth_a[:i + 1, each_test]) + eta_b[:, each_test]).reshape(ytest.shape) for
                i in range(nlatent)]
        ll.append([np.sum(stats.poisson.logpmf(ytest, each)) for each in yhat])

    return np.array(ll)
