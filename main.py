import os.path
from datetime import datetime

from pylab import figure, vlines, subplots, savefig, plot, yticks, xlim, ylim, suptitle, legend, title, gca
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from sklearn.decomposition.factor_analysis import FactorAnalysis

from model_chol import *
from util import likelihood
import simulation

np.random.seed(0)

T = 1000
p = 0
L = 2
N = 50

high = np.log(25 / T)
low = np.log(5 / T)

# simulate latent processes
# x, ticks = simulation.latents(L, T, std, w)
x = np.empty((T, L), dtype=float)
x[:T // 2, 0] = high
x[T // 2:, 0] = low
x[:, 1] = 2 * np.sin(np.linspace(0, 2 * np.pi * 5, T))
for l in range(L):
    x[:, l] -= np.mean(x[:, l])

# simulate spike trains
a = 2 * np.random.rand(L, N) - 1
for l in range(L):
    a[l, :] /= linalg.norm(a[l, :]) / np.sqrt(N)

b = np.empty((1 + p*N, N))
b[0, :] = low

y, _, rate = simulation.spikes(x, a, b, intercept=True)

fa = FactorAnalysis(n_components=L, svd_method='lapack')
m0 = fa.fit_transform(y)
a0 = fa.components_
# a0 = np.random.randn(L, N)
m0 *= np.linalg.norm(a0) / np.sqrt(N)
a0 /= np.linalg.norm(a0) / np.sqrt(N)

var = np.ones(L, dtype=float) * 5
w = np.empty(L, dtype=float)
w[0] = 1e-4
w[1] = 5e-4

# w = np.logspace(-1, 3, 5)
# grid_w = cartesian([w] * L)

control = {'max iteration': 50,
           'fixed-point iteration': 5,
           'tol': 1e-4,
           'verbose': True}

lbound, m1, a1, b1, new_var, new_scale, a0, b0, elapsed, converged = train(y, 0, var, w,
                                                                           a0=a0,
                                                                           b0=None,
                                                                           m0=m0,
                                                                           normofalpha=np.sqrt(N),
                                                                           fixalpha=False, fixbeta=False,
                                                                           fixpostmean=False,
                                                                           hyper=True,
                                                                           kchol=20,
                                                                           control=control)

if not os.path.isdir('output'):
    os.mkdir('output')

it = len(lbound)
dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
log = h5py.File('output/{}.hdf5'.format(dt), 'a')
log.create_dataset(name='iteration', data=it)
log.create_dataset(name='convergence', data=converged)
log.create_dataset(name='time', data=elapsed)
log.create_dataset(name='lower bounds', data=lbound)
log.create_dataset(name='variance', data=var)
log.create_dataset(name='smoothness', data=w)
log.create_dataset(name='true beta', data=b)
log.create_dataset(name='true alpha', data=a)
log.create_dataset(name='spike', data=y)
log.create_dataset(name='latent', data=x)
log.create_dataset(name='initial alpha', data=a0)
log.create_dataset(name='initial beta', data=b0)
log.create_dataset(name='posterior mean', data=m1)
# log.create_dataset(name='posterior covariance', data=V1)
log.create_dataset(name='estimated alpha', data=a1)
log.create_dataset(name='estimated beta', data=b1)
log.close()

with open('output/{}.txt'.format(dt), 'w+') as logging:
    print('{} iteration(s)'.format(it), file=logging)
    print('converged: {}'.format(converged), file=logging)
    print('time: {}s'.format(elapsed), file=logging)
    print('Lower bounds:\n{}'.format(lbound), file=logging)
    print('Posterior mean:\n{}'.format(m1), file=logging)
    # print('Posterior covariance:\n{}'.format(V1), file=logging)
    print('beta: {}'.format(np.linalg.norm(b1 - b)), file=logging)
    print('alpha: {}'.format(np.linalg.norm(a1 - a)), file=logging)
    print('alpha norm: {}'.format(np.linalg.norm(a1)), file=logging)
    print('location of posterior mean: {}'.format(np.mean(m1)), file=logging)
    print('true likelihood: {}'.format(likelihood(y, x, a, b, intercept=True)), file=logging)
    print('estimated likelihood: {}'.format(likelihood(y, m1, a1, b1, intercept=True)), file=logging)
    print('constant rate likelihood: {}'.format(np.sum(y * np.log(y.mean(axis=0)) - y.mean(axis=0))), file=logging)
    print('saturated likelihood: {}'.format(-y.sum()), file=logging)

pp = PdfPages('output/{}.pdf'.format(dt))

_, ax = subplots(N, sharex=True)
for n in range(N):
    ax[n].plot(rate[:, n])
    ax[n].axis('off')
suptitle('Firing rates')
savefig(pp, format='pdf')

# plot spike trains
figure()
ylim(0, N)
for n in range(N):
    vlines(np.arange(T)[y[:, n] > 0], n, n + 1, color='black')
title('{} Spike trains'.format(N))
yticks(range(N))
gca().invert_yaxis()
savefig(pp, format='pdf')

# plot factor analysis
figure()
plot(m0)
savefig(pp, format='pdf')

# plot lowerbound
figure()
frm = 1
plot(range(frm + 1, it + 1), lbound[frm:])
yticks([])
xlim([frm + 1, it + 1])
title('Lower bound={:.3f}, iteration={:d}, time={:.2f}s, L={:d}, N={:d}'.format(lbound[it - 1], it, elapsed, L, N))
savefig(pp, format='pdf')

# plot latent
ns = 500
for l in range(L):
    figure()
    # z = np.random.randn(T, ns)
    # lt = np.linalg.cholesky(V[l, :, :])
    # s = np.dot(lt, z)
    # for n in range(ns):
    #     plot(s[:, n] + m[:, l], color='0.8')
    plot(x[:, l] - np.mean(x[:, l]), label='latent', color='blue')
    plot(m1[:, l], label='posterior', color='red')
    legend()
    title('Latent {}'.format(l + 1))
    savefig(pp, format='pdf')

rotate = np.linalg.lstsq(m1, x)[0]
m2 = np.dot(m1, rotate)
ns = 500
# z = np.random.randn(ns, T, L)
# for l in range(L):
#     z[:, :, l] = np.dot(np.linalg.cholesky(V[l, :, :]), z[:, :, l].T).T + m[:, l]
for l in range(L):
    figure()
    #     # for n in range(ns):
    #     #     plot(np.dot(z[n, :, :], rotate)[:, l], color='0.8')
    plot(x[:, l] - np.mean(x[:, l]), label='latent', color='blue')
    plot(m2[:, l], label='transformed posterior', color='red')
    legend()
    title('Latent (transformed posterior) {}'.format(l + 1))
    savefig(pp, format='pdf')

# _, ax = subplots(L, sharex=True)
# for l in range(L):
#     ax[l].bar(np.arange(N), a[l, :], width=0.25, color='blue', label='true')
#     ax[l].bar(np.arange(N) + 0.25, a1[l, :], width=0.25, color='red', label='estimate')
#     ax[l].axis('off')
# suptitle('alpha')
# savefig(pp, format='pdf')
#
# _, ax = subplots(N, sharex=True)
# for n in range(N):
#     ax[n].bar(np.arange(b.shape[0]), b[:, n], width=0.25, color='blue', label='true')
#     ax[n].bar(np.arange(b.shape[0]) + 0.25, b1[:, n], width=0.25, color='red', label='estimate')
#     ax[n].axis('off')
# suptitle('beta')
# savefig(pp, format='pdf')

pp.close()
