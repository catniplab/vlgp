"""
Convert MAT file to HDF5
"""
import os.path as op
import numpy as np
import h5py

print('Convert MAT file to HDF5')

# paths
samplepath = op.expanduser("~/data/sample")
outputpath = op.expanduser("~/data/output")

# load Graf 5 sample
print('Load MAT')
Graf_5 = h5py.File(op.join(samplepath, 'Graf_5.mat'), 'r')
ori = np.squeeze(np.array(Graf_5['ori']))  # orient of trials
y = np.transpose(np.array(Graf_5['y']), axes=(2, 1, 0))  # transpose y to the shape of trial, time, neuron

# save HDF5
print('Save HDF5')
with h5py.File(op.join(samplepath, 'graf5.h5'), 'w') as hf:
    hf.create_dataset('y', data=y, compression="gzip")  # compress spike trains
    hf.create_dataset('ori', data=ori)
    
print('Done')