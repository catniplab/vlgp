import matplotlib.pyplot as plt

from vb import *
import simulation

dt = 1.0
T = 200
l = 1e-4
sigma = 1
p = 1

L = 1
N = 10
np.random.seed(0)

# simulate latent processes
x, ticks = simulation.latents(L, T, sigma, l)

# simulate spike trains
a = np.random.randn(L, N)  # (L, N)
a /= np.linalg.norm(a)
b = np.random.randn(1 + p * N, N)  # (1 + p*N, N)
b[0, :] = -4
y, Y = simulation.spikes(x, a, b)

# plot spikes
# plt.figure()
# plt.ylim(0, N)
# for n in range(N):
#     plt.vlines(np.arange(T)[y[:, n] > 0], n, n + 1, color='black')
# plt.title('Spike')
# plt.yticks([])
# plt.xlim([0, T])

# mu = np.random.randn(T, L) + 1
# mu = x
mu = np.zeros((T, L))
cov = np.empty((T, T))
for i, j in itertools.product(range(T), range(T)):
    cov[i, j] = simulation.sqexp(i - j, l)
sigma = np.zeros((L, T, T))
for l in range(L):
    sigma[l, :, :] = cov + np.identity(T) * 1e-7

# print 'Prior mean\n', mu
# print 'Prior covariance', sigma
# b[0, :] = -10
m, V, b1, a1, lbound, it, elapsed = variational(y, mu, sigma, p,
                                                a0=np.random.randn(L, N),
                                                b0=np.random.randn(1 + p * N, N),
                                                m0=mu,
                                                V0=sigma,
                                                r=np.finfo(float).eps, maxiter=500, inneriter=3, tol=0.05,
                                                verbose=True)

print '%d iteration(s)' % it
print 'time: %.3fs' % elapsed
print 'Lower bounds:\n', lbound[:it]
# print 'Posterior mean\n', m
print 'covariance:\n', V
print 'beta:\n', b
print 'alpha:\n', a


plt.figure()
plt.plot(lbound[1:])
plt.title('Lower bound (%d iterations)' % it)
ns = 100
for l in range(L):
    # for _ in range(5):
    #     sample = np.random.multivariate_normal(m[:, l], V[l, :, :])
    #     plt.plot(sample, color='.8')
    plt.figure()
    z = np.random.randn(T, ns)
    lt = np.linalg.cholesky(V[l, :, :])
    s = np.dot(lt, z)
    for n in range(ns):
        plt.plot(s[:, n] + m[:, l], color='0.8')
    plt.plot(x[:, l], label='latent', color='blue')
    plt.plot(-x[:, l], label='negative latent', color='green')
    plt.plot(m[:, l], label='posterior', color='red')
    plt.legend()
    plt.title('Latent and Posterior')
    # plt.figure()
    # plt.plot(-m[:, l], label='Posterior', color='red')
    # plt.title('Posterior (neg)')
    # plt.title('Posterior mean and latent')
    # plt.legend()

plt.show()
