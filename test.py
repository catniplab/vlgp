import itertools
import os.path
import matplotlib.pyplot as plt
from vb import *
import simulation

dt = 1.0
T = 200
l = 1e-4
std = 5
p = 1

L = 1
N = 10
np.random.seed(0)

# simulate latent processes
x, ticks = simulation.latents(L, T, std, l)
x -= 3
# x[:100] +=1
# mean = np.ones(x.shape)
# x += mean

# simulate spike trains
a = np.ones((L, N))  # (L, N)
a /= np.linalg.norm(a)
b = 0 * np.ones((1 + p * N, N))  # (1 + p*N, N)
b[0, :] = 0
y, Y = simulation.spikes(x, a, b)

# plot spikes
plt.figure()
plt.ylim(0, N)
for n in range(N):
    plt.vlines(np.arange(T)[y[:, n] > 0], n, n + 1, color='black')
plt.title('Spike trains')
plt.yticks([])
plt.xlim([0, T])

# mu = np.random.randn(T, L) + 1
# mu = x
mu = 0 * np.ones((T, L))
cov = np.empty((T, T))
for i, j in itertools.product(range(T), range(T)):
    cov[i, j] = 10 * simulation.sqexp(i - j, l)
sigma = np.zeros((L, T, T))
for l in range(L):
    sigma[l, :, :] = cov + np.identity(T) * 1e-7

# print 'Prior mean\n', mu
# print 'Prior covariance', sigma
# b[0, :] = -10
intercept = True
m, V, b1, a1, lbound, elapsed, convergent = variational(y, mu, sigma, p,
                                                        # a0=a + 1,
                                                        # b0=None,
                                                        a0=a,
                                                        b0=b,
                                                        m0=mu,
                                                        V0=sigma,
                                                        intercept=intercept,
                                                        maxiter=200, inneriter=5, tol=1e-4,
                                                        fixed=True, constraint=False,
                                                        verbose=True)

it = len(lbound)
num= time.time()
if not os.path.isdir('output'):
    os.mkdir('output')
with open('output/[%d] L=%d N=%d.txt' % (num, L, N), 'w+') as logging:
    print('{} iteration(s)'.format(it), file=logging)
    print('convergent: {}'.format(convergent), file=logging)
    print('time: {}s'.format(elapsed), file=logging)
    print('Lower bounds:\n{}'.format(lbound), file=logging)
    print('Posterior mean:\n{}'.format(m), file=logging)
    print('Posterior covariance:\n{}'.format(V), file=logging)
    # print('beta: {}'.format(np.linalg.norm(b1 - b)), file=logging)
    print('alpha: {}'.format(np.linalg.norm(a1 - a)), file=logging)
    # print('true likelihood: {}'.format(likelihood(y, x, a, b, intercept=intercept)), file=logging)
    # print('estimated likelihood: {}'.format(likelihood(y, m, a1, b1, intercept=intercept)), file=logging)
    # print('constant rate likelihood: {}'.format(np.sum(y * np.log(y.mean(axis=0)) - y.mean(axis=0))),
    #       file=logging)
    # print('saturated likelihood: {}'.format(-y.sum()), file=logging)

plt.figure()
frm = 1
plt.plot(range(frm + 1, it + 1), lbound[frm:])
plt.yticks([])
plt.xlim([frm + 1, it + 1])
title = '[%d] Lower bound=%.3f, iteration=%d, time=%.2fs, L=%d, N=%d' % (num, lbound[it-1], it, elapsed, L, N)
plt.title(title)
plt.savefig('output/{}.png'.format(title))
ns = 500
for l in range(L):
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
    title = '[%d] Latent %d N=%d' % (num, l + 1, N)
    plt.title(title)
    plt.savefig('output/{}.png'.format(title))

for l in range(L):
    plt.figure()
    plt.bar(np.arange(N), a[l, :], width=0.25, color='blue', label='true')
    plt.bar(np.arange(N) + 0.25, a1[l, :], width=0.25, color='red', label='estimate')
    plt.legend()
    title = '[%d] alpha[%d]' % (num, l)
    plt.title(title)

plt.show()
