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

vLGP_4D = loadmat(op.join(outputpath, 'Graf_vLGP_4D.mat'), squeeze_me=True)['Graf_vLGP_4D']
omega = vLGP_4D['omega'].tolist()
print(omega)

graf5 = pickle.load(open(op.join(outputpath, 'graf5SpikeByOri.dat'), 'rb'))

uniqOri = np.linspace(0, 360, 72, endpoint=False)

y = np.concatenate((graf5['y'][uniqOri == 0, :], graf5['y'][uniqOri == 90, :]))

print(y.shape)