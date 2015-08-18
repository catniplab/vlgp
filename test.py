import os.path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp
from vb import *
import simulation

dt = 1.0
T = 500
l = 1e-4
std = 2
p = 1

L = 1
N = 25
np.random.seed(0)

high = np.log(5/100)
low = np.log(0.1/100)

# simulate latent processes
x, ticks = simulation.latents(L, T, std, l)
x[:T//2, :] = high
x[T//2:, :] = low

# simulate spike trains
a = np.random.choice([-1, 1], size=(L, N))
# a /= np.linalg.norm(a)
b = 0 * np.ones((p * N, N))
c = np.diag(np.dot(a.T, (a < 0) * -(high + low)))
y, Y, rate = simulation.spikes2(x, a, b, c)

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

a0 = a + np.random.randn(*a.shape)
a0 /= np.linalg.norm(a0) / 5
m, V, a1, b1, c1, lbound, elapsed, convergent = variational(y, mu, sigma, p,
                                                            a0=a0,
                                                            b0=b,
                                                            c0=None,
                                                            # c0=c,
                                                            # a0=a0,
                                                            # b0=None,
                                                            m0=mu,
                                                            V0=sigma,
                                                            fixa=False, fixb=True, fixc=False, fixm=False, fixV=False,
                                                            anorm=5,
                                                            maxiter=200, inneriter=5, tol=1e-4,
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

fig, ax = plt.subplots(N, sharex=True)
for n in range(N):
    ax[n].plot(rate[:, n])
    ax[n].axis('off')
plt.suptitle('Firing rates')
plt.savefig(pp, format='pdf')

# plot spike trains
plt.figure()
plt.ylim(0, N)
for n in range(N):
    plt.vlines(np.arange(T)[y[:, n] > 0], n, n + 1, color='black')
plt.title('{} Spike trains'.format(N))
plt.yticks(range(N))
plt.gca().invert_yaxis()
plt.savefig(pp, format='pdf')

# plot lowerbound
plt.figure()
frm = 1
plt.plot(range(frm + 1, it + 1), lbound[frm:])
plt.yticks([])
plt.xlim([frm + 1, it + 1])
plt.title('Lower bound={:.3f}, iteration={:d}, time={:.2f}s, L={:d}, N={:d}'.format(lbound[it-1], it, elapsed, L, N))
plt.savefig(pp, format='pdf')

# plot latent
ns = 500
for l in range(L):
    plt.figure()
    z = np.random.randn(T, ns)
    lt = np.linalg.cholesky(V[l, :, :])
    s = np.dot(lt, z)
    for n in range(ns):
        plt.plot(s[:, n] + m[:, l], color='0.8')
    plt.plot(x[:, l], label='latent', color='blue')
    plt.plot(m[:, l], label='posterior', color='red')
    plt.legend()
    plt.title('Latent {}'.format(l + 1))
    plt.savefig(pp, format='pdf')

fig, ax = plt.subplots(L, sharex=True)
for l in range(L):
    ax.bar(np.arange(N), a[l, :], width=0.25, color='blue', label='true')
    ax.bar(np.arange(N) + 0.25, a1[l, :], width=0.25, color='red', label='estimate')
    ax.axis('off')
plt.suptitle('alpha')
plt.savefig(pp, format='pdf')

fig, ax = plt.subplots(N, sharex=True)
for n in range(N):
    ax[n].bar(np.arange(b.shape[0]), b[:, n], width=0.25, color='blue', label='true')
    ax[n].bar(np.arange(b.shape[0]) + 0.25, b1[:, n], width=0.25, color='red', label='estimate')
    ax[n].axis('off')
plt.suptitle('beta')
plt.savefig(pp, format='pdf')

plt.figure()
plt.bar(np.arange(N), c, width=0.25, color='blue', label='true')
plt.bar(np.arange(N) + 0.25, c1, width=0.25, color='red', label='estimate')
plt.xticks([])
plt.legend()
plt.title('gamma')
plt.savefig(pp, format='pdf')

pp.close()