"""
Stateful callbacks
"""
import logging
import time

from .util import save

logger = logging.getLogger(__name__)


class Saver:
    def __init__(self):
        self.last_saving_time = time.perf_counter()

    def save(self, trials, params, config, force=False):
        now = time.perf_counter()
        if force or now - self.last_saving_time > config['saving_interval']:
            logger.info('Saving model to {}'.format(config['path']))
            save({'trials': trials, 'params': params, 'config': config})
            self.last_saving_time = time.perf_counter()
