"""constants"""

# coding observation dimensions
from collections import defaultdict

from .name import Z_DIM, Y

NA = 0  # dimension not to use
POISSON = 1  # spike train
GAUSSIAN = 2  # local field potential

LIK_CODE = defaultdict(int,
                       poisson=POISSON,
                       gaussian=GAUSSIAN)

REQUIRED_FIELDS = [Y, Z_DIM]

# default options
DEFAULT_VALUES = {'learning_rate': 1.0,
                  'niter': 50,  # max outer iterations
                  'tol': 1e-4,  # tolerance
                  'eps': 1e-8,  # small quantity
                  'initialize': 'fa',
                  # posterior
                  'learn_post': True,
                  'e_niter': 10,
                  'method': 'VB',
                  'constrain_mu': 'both',
                  'dmu_bound': 1.0,
                  # parameters
                  'learn_param': True,
                  'm_niter': 10,
                  'hessian': True,
                  'constrain_a': False,
                  'da_bound': 1.0,
                  'db_bound': 1.0,
                  # GP
                  'learn_hyper': True,
                  'nhyper': 2,
                  'hyper_obj': 'ELBO',
                  'gp_noise': 1e-4,
                  'omega_bound': (1e-5, 1e-3),
                  'seg_len': 20,
                  'gp': 'cutting',
                  'sigma': 1,
                  'tau': 100,
                  # misc
                  'verbose': False,
                  'saving_interval': 3600}

