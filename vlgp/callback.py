import logging
import time
from pprint import pprint

from .util import save


logger = logging.getLogger(__name__)


class Saver:
    def __init__(self):
        self.last_saving_time = time.perf_counter()

    def save(self, model, force=False):
        now = time.perf_counter()
        if force or now - self.last_saving_time > model['saving_interval']:
            logger.info('Saving model to {}'.format(model['path']))
            save(model, model['path'])
            self.last_saving_time = time.perf_counter()


class Progressor:
    def __init__(self, total):
        try:
            from ipywidgets import FloatProgress
            from IPython.display import display
            self.pbar = FloatProgress(min=0, max=total)
            display(self.pbar)
        except ImportError:
            self.pbar = dict(value=0)

    def update(self, model):
        self.pbar.value += 1


class Printer:
    @staticmethod
    def print(model):
        stat = dict()
        stat['it'] = model['it']
        stat['E-step'] = model['e_elapsed'] and model['e_elapsed'][-1]
        stat['M-step'] = model['m_elapsed'] and model['m_elapsed'][-1]
        stat['H-step'] = model['h_elapsed'] and model['h_elapsed'][-1]
        stat['sigma'] = model['sigma']
        stat['omega'] = model['omega']
        if model['verbose']:
            pprint(stat, indent=4)
