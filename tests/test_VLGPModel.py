from unittest import TestCase

import numpy as np
from scipy.linalg import toeplitz

import vlgp
from vlgp import simulation as sim


def gp(theta, std, nbin, dim):
    eps = 1e-10
    dsq = np.arange(nbin) ** 2
    k = np.exp(- theta * dsq)
    K = toeplitz(k) + eps * np.identity(nbin)
    x = std * np.random.multivariate_normal(np.zeros(nbin), K, dim).T
    x -= x.mean(axis=0)
    return x


class TestVLGPModel(TestCase):
    def test_fit(self):
        np.random.seed(0)
        latent_dim = 2
        bias = np.log(20 / 100)  # log base firing rate
        theta = 1e-2
        std = 1.0
        sigma = np.full(latent_dim, fill_value=std)
        omega = np.full(latent_dim, fill_value=theta)
        nbin = 200
        ntrial = 2
        neuron_dim = 100

        x = gp(theta, std, nbin * ntrial, latent_dim)
        x -= x.mean(axis=0, keepdims=True)

        a = 0.5 * np.sort(
            (np.random.rand(latent_dim, neuron_dim) + 1) * np.sign(np.random.randn(latent_dim, neuron_dim)),
            axis=1)  # loading matrix
        b = np.vstack(
            (bias * np.ones(neuron_dim), -10 * np.ones(neuron_dim), -10 * np.ones(neuron_dim)))  # regression weights
        y, _, _ = sim.spike(x, a, b)
        y = y.reshape((-1, nbin, neuron_dim))[:ntrial, ...]

        model = vlgp.VLGPModel(latent_dim, neuron_dim, nbin, 2)
        model.fit(y, ['spike'] * y.shape[-1], sigma, omega, nbin, fixed_prior_scale=True, fixed_prior_std=True)
