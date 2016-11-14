import time
from pprint import pprint

from tqdm import tqdm

from .util import save


class Callback:
    def __init__(self, model):
        self.model = model


class Saver(Callback):
    def __init__(self, model):
        super().__init__(model)
        self.period = self.model['options']['saving_interval']
        self.last_saving_time = time.perf_counter()

    def __call__(self, *args, **kwargs):
        now = time.perf_counter()
        if now - self.last_saving_time > self.period:
            print('Saving')
            save(self.model, self.model['path'])
            self.last_saving_time = now
            print('Saved')


class Progress(Callback):
    def __init__(self, model):
        super().__init__(model)
        self.pbar = tqdm(total=self.model['options']['niter'])

    def __call__(self, *args, **kwargs):
        options = self.model['options']

        self.pbar.update(1)

        stat = dict()
        stat['E-step'] = self.model['e_elapsed'][-1]
        stat['M-step'] = self.model['m_elapsed'][-1]
        stat['H-step'] = self.model['h_elapsed'][-1]
        stat['sigma'] = self.model['sigma']
        stat['omega'] = self.model['omega']

        if options['verbose']:
            pprint(stat)

    def __del__(self):
        self.pbar.close()
