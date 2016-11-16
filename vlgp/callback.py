import time
from pprint import pprint

from tqdm import tqdm

from .util import save


class Saver():
    def __init__(self):
        self.last_saving_time = time.perf_counter()

    def save(self, model, force=False):
        now = time.perf_counter()
        if force or now - self.last_saving_time > model['options']['saving_interval']:
            print('Saving model to {}'.format(model['path']))
            save(model, model['path'])
            self.last_saving_time = time.perf_counter()
            print('Model saved')


class Progressor():
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def update(self, model):
        self.pbar.update(1)
        self.print(model)

    def print(self, model):
        self.pbar.update(0)
        options = model['options']
        stat = dict()
        stat['E-step'] = model['e_elapsed'][-1]
        stat['M-step'] = model['m_elapsed'][-1]
        stat['H-step'] = model['h_elapsed'][-1]
        stat['sigma'] = model['sigma']
        stat['omega'] = model['omega']
        if options['verbose']:
            pprint(stat)

    def __del__(self):
        self.pbar.close()
