from unittest import TestCase

import numpy as np

import vlgp
from vlgp.simulation import gp_mvn, spike


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

        x = gp_mvn(theta, std, nbin * ntrial, latent_dim)
        x -= x.mean(axis=0, keepdims=True)

        a = 0.5 * np.sort(
            (np.random.rand(latent_dim, neuron_dim) + 1) * np.sign(np.random.randn(latent_dim, neuron_dim)),
            axis=1)  # loading matrix
        b = np.vstack(
            (bias * np.ones(neuron_dim), -10 * np.ones(neuron_dim), -10 * np.ones(neuron_dim)))  # regression weights
        y, _, _ = spike(x, a, b)
        y = y.reshape((-1, nbin, neuron_dim))[:ntrial, ...]

        model = vlgp.VLGPModel(latent_dim, neuron_dim, nbin, 2)
        model.fit(y, ['spike'] * y.shape[-1], sigma, omega, nbin, fixed_prior_scale=True, fixed_prior_std=True)
