__author__ = 'yuan'

import numpy as np
import matplotlib.pyplot as plt
from vb import *
import simulation
from plot import *


dt = 1.0
T = 500
b = 0.0001
sigma = 1.0
p = 1

L = 1
N = 3
np.random.seed(0)

x, ticks = simulation.latents(L, T, sigma, b)
rate = -3 * np.ones(N, dtype=float)
A = 0.7 * np.ones((N, L), dtype=float)
B = np.dstack((0.5 * np.eye(N), -0.25 * np.eye(N)))
y = simulation.multi_spike(rate, x, A, B)

mu = 0.1*np.ones((T, L), dtype=float)
Sigma = np.zeros((L, T, T))
for t in range(T):
    Sigma[:, t, t] = np.ones(L)


m, V, coeffs, lbound, it = variational(y, mu, Sigma, p, maxiter=10, epsilon=1e-7)
print '%d iteration(s)' % it
print 'Lower bounds: ', lbound[:it]
print 'Coefficients:\n', coeffs
print 'Posterior mean:\n', m
