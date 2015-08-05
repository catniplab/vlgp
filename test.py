__author__ = 'yuan'
import matplotlib.pyplot as plt

from vb import *
import simulation

dt = 1.0
T = 500
b = 1e-3
sigma = 2.0
p = 1

L = 1
N = 3
np.random.seed(0)

# simulate latent processes
x, ticks = simulation.latents(L, T, sigma, b)

# simulate spike trains
a = np.ones((L, N))  # (L, N)
b = np.zeros((1 + p * N, N))  # (1 + p*N, N)
b[0, :] = -1
y, Y = simulation.spikes(x, a, b)
# print y
# print Y

mu = np.zeros((T, L))

cov = np.empty((T, T))
for i, j in itertools.product(range(T), range(T)):
    cov[i, j] = simulation.sqexp(i - j, 1e-3)
sigma = np.zeros((L, T, T))
for l in range(L):
    sigma[l, :, :] = cov

# print 'Prior mean\n', mu
# print 'Prior covariance', sigma
b[0, :] = -10
m, V, b, a, lbound, it = variational(y, mu, sigma, p, b0=b, a0=a, maxiter=50, inneriter=5, epsilon=1e-5, verbose=True)
print '%d iteration(s)' % it
print 'Lower bounds: ', lbound[:it]
# print 'Posterior mean\n', m
print 'Posterior covariance\n', V
# print 'beta\n', b
# print 'alpha\n', a
plt.plot(lbound)
# for l in range(L):
#     plt.figure()
#     plt.plot(x[:, l])
#     plt.plot(m[:, l])
plt.show()
