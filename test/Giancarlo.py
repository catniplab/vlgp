__author__ = 'yuan'
import numpy as np
from scipy import linalg
from scipy.io import loadmat
from pylab import *
from sklearn.decomposition.factor_analysis import FactorAnalysis
import model_chol, decompV

np.random.seed(0)
matfile = loadmat('/Users/yuan/Documents/MATLAB/x')
y = matfile['x'][:10000, 50:]
T, N = y.shape
L = 3
fa = FactorAnalysis(n_components=L, svd_method='lapack')
m0 = fa.fit_transform(y)
a0 = fa.components_
for l in range(L):
    m0[:, l] -= np.mean(m0[:, l])
    m0[:, l] *= np.linalg.norm(a0[l, :]) / np.sqrt(N)
    a0[l, :] /= np.linalg.norm(a0[l, :]) / np.sqrt(N)

var = 5 * np.ones(L, dtype=float)
w = np.ones(L, dtype=float)
w[0] = 1e-8
w[1] = 1e-6
w[2] = 1e-4

lbound, m1, a1, b1, new_var, new_scale, a0, b0, elapsed, converged = decompV.train(y, 0, var, w, b0=None, m0=m0,
                                                                                   anorm=np.sqrt(N), hyper=False,
                                                                                   kchol=100, niter=50, tol=1e-5,
                                                                                   verbose=True)

figure()
plot(m1)
show()

