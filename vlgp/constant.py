"""constants"""

from collections import defaultdict

# names
VB = 'VB'
MAP = 'MAP'
X = 'x'
Z = 'z'
MU = 'mu'
W = 'w'
V = 'v'
dMU = 'dmu'
dA = 'da'
dB = 'db'
ESTEP = 'learn_post'
MSTEP = 'learn_param'
HSTEP = 'learn_hyper'
HPERIOD = 'nhyper'
HOBJ = 'hyper_obj'
TRIALLET = 'subsample_size'
PRIOR = 'chol'
PRIORSTD = 'sigma'
PRIORTIMESCALE = 'omega'
MAXITER = 'niter'
ITER = 'it'
LIK = 'lik'
Y_DIM = 'y_dim'
#

NA = 0  # dimension not to use
POISSON = 1
GAUSSIAN = 2

LIK_CODE = defaultdict(int,
                       poisson=POISSON,
                       gaussian=GAUSSIAN)

Y = 'y'
Z_DIM = 'lat_dim'
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


