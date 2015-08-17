import os.path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp
from vb import *
import simulation

dt = 1.0
T = 500
l = 1e-4
std = 5
p = 1

L = 1
N = 5
np.random.seed(0)

# simulate latent processes
x, ticks = simulation.latents(L, T, std, l)
x[:250, :] = np.log(10/T)
x[250:, :] = np.log(0.1/T)

# simulate spike trains
a = np.random.choice([-1, 1], size=(L, N))
# a /= np.linalg.norm(a)
b = 0 * np.ones((p * N, N))
c = (a < 0) * -(np.log(10/T) + np.log(0.1/T))
y, Y, rate = simulation.spikes2(x, a, b, c)

plt.figure()
for n in range(N):
    plt.subplot(N, 1, n + 1)
    plt.plot(rate[:, n])
    plt.ylim([-0.005, 0.05])


# mu = np.random.randn(T, L) + 1
# mu = x
mu = 0 * np.ones((T, L))
cov = np.empty((T, T))
for i, j in itertools.product(range(T), range(T)):
    cov[i, j] = 10 * simulation.sqexp(i - j, l)
sigma = np.zeros((L, T, T))
for l in range(L):
    sigma[l, :, :] = cov + np.identity(T) * 1e-7
    # sigma[l, :, :] = 0.2 * np.identity(T)

a0 = np.abs(np.random.randn(*a.shape))
a0 /= np.linalg.norm(a0)
m, V, a1, b1, c1, lbound, elapsed, convergent = variational(y, mu, sigma, p,
                                                            a0=a0,
                                                            b0=b,
                                                            # c0=c,
                                                            # a0=a0,
                                                            # b0=None,
                                                            m0=mu,
                                                            V0=sigma,
                                                            fixa=False, fixb=True, fixc=False, fixm=False, fixV=False,
                                                            constraint=False,
                                                            maxiter=200, inneriter=5, tol=1e-5,
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
    print('beta: {}'.format(np.linalg.norm(b1 - b)), file=logging)
    print('alpha: {}'.format(np.linalg.norm(a1 - a)), file=logging)
    print('gamma: {}'.format(np.linalg.norm(c1 - c)), file=logging)
    print('true likelihood: {}'.format(likelihood(y, x, a, b, c)), file=logging)
    print('estimated likelihood: {}'.format(likelihood(y, m, a1, b1, c1)), file=logging)
    print('constant rate likelihood: {}'.format(np.sum(y * np.log(y.mean(axis=0)) - y.mean(axis=0))), file=logging)
    print('saturated likelihood: {}'.format(-y.sum()), file=logging)

pp = PdfPages('output/[{:.0f}].pdf'.format(num))

# plot spike trains
plt.figure()
plt.ylim(0, N)
for n in range(N):
    plt.vlines(np.arange(T)[y[:, n] > 0], N - n, N - n - 1, color='black')
title = 'Spike trains'
plt.title(title)
plt.yticks([])
# plt.xlim([0, T])
# plt.savefig('output/{}.png'.format(title))
plt.savefig(pp, format='pdf')

# plot lowerbound
plt.figure()
frm = 1
plt.plot(range(frm + 1, it + 1), lbound[frm:])
plt.yticks([])
plt.xlim([frm + 1, it + 1])
title = 'Lower bound=%.3f, iteration=%d, time=%.2fs, L=%d, N=%d' % (lbound[it-1], it, elapsed, L, N)
plt.title(title)
# plt.savefig('output/{}.png'.format(title))
plt.savefig(pp, format='pdf')

ns = 500
for l in range(L):
    plt.figure()
    z = np.random.randn(T, ns)
    lt = np.linalg.cholesky(V[l, :, :])
    s = np.dot(lt, z)
    for n in range(ns):
        plt.plot(s[:, n] + m[:, l], color='0.8')
    plt.plot(x[:, l], label='latent', color='blue')
    # plt.plot(-x[:, l], label='negative latent', color='green')
    plt.plot(m[:, l], label='posterior', color='red')
    # plt.ylim([-15, 0])
    plt.legend()
    title = 'Latent %d N=%d' % (l + 1, N)
    plt.title(title)
    # plt.savefig('output/{}.png'.format(title))
    plt.savefig(pp, format='pdf')

plt.figure()
title = 'alpha'
plt.title(title)
# plt.legend()
for l in range(L):
    plt.subplot(L, 1, l + 1)
    plt.bar(np.arange(N), a[l, :], width=0.25, color='blue', label='true')
    plt.bar(np.arange(N) + 0.25, a1[l, :], width=0.25, color='red', label='estimate')
    plt.xticks([])
plt.savefig(pp, format='pdf')
# plt.savefig('output/{}.png'.format(title))

plt.figure()
title = 'beta'
plt.title(title)
# plt.legend()
for n in range(N):
    plt.subplot(N, 1, n + 1)
    plt.bar(np.arange(b.shape[0]), b[:, n], width=0.25, color='blue', label='true')
    plt.bar(np.arange(b.shape[0]) + 0.25, b1[:, n], width=0.25, color='red', label='estimate')
    plt.xticks([])
# plt.savefig('output/{}.png'.format(title))
plt.savefig(pp, format='pdf')



pp.close()