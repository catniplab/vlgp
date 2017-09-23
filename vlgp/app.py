import os

import fire
import h5py
import numpy as np

from vlgp import api


class App:
    def fit(self, infile, lat_dim, seed=None, outfile=None, **kwargs):
        np.random.seed(seed)

        infile = os.path.abspath(infile)
        outfile = outfile or os.getcwd() + '/vlgp_fit.h5'
        outfile = os.path.abspath(outfile)

        print('Loading data from', infile)

        with h5py.File(infile, 'r') as f:
            y = np.asarray(f['y'])
            lik = np.asarray(f['lik'])

        print('Data loaded')

        print('Fitting')

        api.fit(y=y, lik=lik, path=outfile, lat_dim=lat_dim, **kwargs)

        print('Done')


if __name__ == '__main__':
    fire.Fire(App)
