import os.path as op
import numpy as np
import scipy as sp
from scipy.special import expit
from scipy import stats
from scipy import linalg
from scipy.io import loadmat, savemat
from scipy.linalg import orth, svd
from numpy.linalg import norm
from numpy import dstack, rollaxis
from sklearn.decomposition.factor_analysis import FactorAnalysis
import h5py
import pickle

import simulation, util, graph, hyper, vlgp
from mathf import ichol_gauss, subspace
from util import rotate, add_constant


samplepath = op.expanduser("~/data/sample")
outputpath = op.expanduser("~/data/output")
figurepath = op.expanduser("~/data/figure")

with h5py.File(op.join(samplepath, 'graf5.h5'), 'r') as hf:
    y = np.array(hf['y'])
    ori = np.array(hf['ori'])
    idx = np.logical_or(ori == 0, ori == 90)
    
ori0and90 = ori[idx]

y0and90 = y[idx.squeeze(), :]

nlatent = 4 # 4D latent
np.random.seed(0)
sigma = np.full(nlatent, fill_value=3.0)
omega = np.full(nlatent, fill_value=1e-4)
fit0and90 = vlgp.fit(y0and90, ['spike'] * y0and90.shape[-1], sigma, omega, lag=2, rank=100, adjhess=True, decay=0,
               niter=100, tol=1e-5, verbose=True, learn_sigma=True, learn_omega=True, nhyper=5)
               
with open(op.join(outputpath, 'graf5Hyper3v.h5'), 'wb') as outfile:
    pickle.dump(fit0and90, outfile, protocol=pickle.HIGHEST_PROTOCOL)
