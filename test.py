__author__ = 'yuan'
import matplotlib.pyplot as plt

from vb import *
import simulation

dt = 1.0
T = 200
l = 1e-4
sigma = 1
p = 1

L = 1
N = 3
np.random.seed(0)

# simulate latent processes
x, ticks = simulation.latents(L, T, sigma, l)

# simulate spike trains
a0 = np.random.randn(L, N)  # (L, N)
b0 = np.random.randn(1 + p * N, N)  # (1 + p*N, N)
b0[0, :] = -3
y, Y = simulation.spikes(x, a0, b0)

# plot spikes
plt.figure()
plt.ylim(0, N)
for n in range(N):
    plt.vlines(np.arange(T)[y[:, n] > 0], n, n + 1, color='black')
plt.title('Spike')
plt.yticks([])

# mu = np.random.randn(T, L) + 1
mu = np.zeros((T, L)) + 1
cov = np.empty((T, T))
for i, j in itertools.product(range(T), range(T)):
    cov[i, j] = simulation.sqexp(i - j, l)
sigma = np.zeros((L, T, T))
for l in range(L):
    sigma[l, :, :] = cov + np.eye(T)

# print 'Prior mean\n', mu
# print 'Prior covariance', sigma
# b[0, :] = -10
m, V, b, a, lbound, it = variational(y, mu, sigma, p,
                                     b0=b0 + 1,
                                     a0=a0 + 1,
                                     stepsize=5, maxiter=500, inneriter=3, epsilon=1e-7, verbose=True)
print '%d iteration(s)' % it
print 'Lower bounds: ', lbound[:it]
# print 'Posterior mean\n', m
print 'covariance: %d' % np.linalg.norm(V - sigma)
print 'beta: %d' % np.linalg.norm(b - b0)
print 'alpha: %d' % np.linalg.norm(a - a0)



plt.figure()
plt.plot(lbound)
plt.title('Lower bound (%d iterations)' % it)
for l in range(L):
    plt.figure()
    plt.plot(x[:, l], label='Latent value')
    plt.plot(m[:, l], label='Posterior mean')
    plt.title('Posterior mean and latent')
    plt.legend()
plt.show()
