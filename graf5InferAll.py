import os.path as op
import pickle
import h5py
import numpy as np
from scipy.io import loadmat

import vlgp

samplepath = op.expanduser("~/data/sample")
outputpath = op.expanduser("~/data/output")
figurepath = op.expanduser("~/data/figure")


print('Reading Sample')
Graf_5 = h5py.File(op.join(samplepath, 'Graf_5.mat'), 'r')

orient = np.squeeze(np.array(Graf_5['ori']))
unique_ori = np.unique(orient)
y = np.array(Graf_5['y'])
# transpose y (into trial, time, neuron)
y = np.transpose(y, axes=(2, 1, 0))

print('Sample: {}'.format(y.shape))
print('Orient: {}'.format(orient))

print('Reading omega')
vLGP_4D = loadmat(op.join(outputpath, 'Graf_vLGP_4D.mat'), squeeze_me=True)['Graf_vLGP_4D']
omega = vLGP_4D['omega'].tolist()
print('omega: {}'.format(omega))

print('Reading parameter')
fit_0_90_4D = loadmat(op.join(outputpath, 'Graf_5_0_90_vLGP_4D_2'), squeeze_me=True)

print('Start inferring')
np.random.seed(0)
sigma = np.full(4, fill_value=2.0)

for ori in unique_ori:
	try:
	    print('Ori: {}'.format(ori))
	    fit = vlgp.fit(y[orient == ori, :], ['spike'] * y.shape[-1], sigma, omega, a=fit_0_90_4D['a'], b=fit_0_90_4D['b'],
	               lag=2, rank=100, niter=50, tol=1e-5, verbose=False, infer='posterior', 
	               learning_rate=1, learn_sigma=False, learn_omega=False, adjhess=True, decay=0, MAP=False)
	    with open(op.join(outputpath, 'graf5Sigma2Deg{:d}.dat'.format(int(ori))), 'wb') as outfile:
	        pickle.dump(fit, outfile, protocol=pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('{:d} degree failed'.format(ori))
print('Inference ends')

print('Concatenating latent')
mu = []
v = []
for ori in unique_ori:
    fit = pickle.load(open(op.join(outputpath, 'graf5Sigma2Deg{:d}.dat'.format(int(ori))), 'rb'))
    mu.append(fit['mu'])
    v.append(fit['v'])
    
latent = np.concatenate(mu)
var = np.concatenate(v)
print(latent.shape, var.shape)
with open(op.join(outputpath, 'graf5Sigma2.dat'), 'wb') as outfile:
    pickle.dump(latent, outfile, protocol=pickle.HIGHEST_PROTOCOL)
with open(op.join(outputpath, 'graf5Sigma2var.dat'), 'wb') as outfile:
    pickle.dump(var, outfile, protocol=pickle.HIGHEST_PROTOCOL)

print('Finished')