import time
from pprint import pprint

# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

from .util import save


class Saver():
    def __init__(self):
        self.last_saving_time = time.perf_counter()

    def save(self, model, force=False):
        now = time.perf_counter()
        if force or now - self.last_saving_time > model['options']['saving_interval']:
            print('\nSaving model to {}'.format(model['path']))
            save(model, model['path'])
            self.last_saving_time = time.perf_counter()
            print('Model saved')


class Progressor():
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def update(self, model):
        self.pbar.update(1)
        if model['options']['verbose']:
            self.print(model)

    def print(self, model):
        self.pbar.update(0)
        options = model['options']
        stat = dict()
        stat['E-step'] = model['e_elapsed'] and model['e_elapsed'][-1]
        stat['M-step'] = model['m_elapsed'] and model['m_elapsed'][-1]
        stat['H-step'] = model['h_elapsed'] and model['h_elapsed'][-1]
        stat['sigma'] = model['sigma']
        stat['omega'] = model['omega']
        # stat['dmu'] = model['dmu']
        # stat['da'] = model['da']
        # stat['db'] = model['db']
        if options['verbose']:
            pprint(stat)

    def __del__(self):
        self.pbar.close()
