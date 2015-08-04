__author__ = 'yuan'
from vb import *
import simulation

dt = 1.0
T = 1000
b = 0.0001
sigma = 1.0
p = 1

L = 1
N = 3
np.random.seed(0)

# simulate latent processes
x, ticks = simulation.latents(L, T, sigma, b)

# simulate spike trains
a = np.random.randn(L, N)  # (L, N)
b = np.random.randn(1 + p * N, N)  # (1 + p*N, N)
b[0, :] = -2
y, Y = simulation.spikes(x, a, b)
# print y
# print Y

mu = np.random.randn(T, L)
sigma = np.zeros((L, T, T))
for l in range(L):
    sigma[l, :, :] = np.eye(T)

m, V, b, a, lbound, it = variational(y, mu, sigma, p, maxiter=20, epsilon=1e-7, verbose=True)
print '%d iteration(s)' % it
print 'Lower bounds: ', lbound[:it]
