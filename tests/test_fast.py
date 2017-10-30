from unittest import TestCase

import numpy as np


class TestFast(TestCase):
    def test_clip_grad(self):
        from vlgp import fast
        np.random.seed(0)
        n = 100
        x = np.random.randn(n)
        x_clipped = fast.clip_grad(x, bound=1.0)

        self.assertTrue(np.all(np.logical_and(x_clipped >= -1.0, x_clipped <= 1.0)))

    def test_cut_trial(self):
        from vlgp import fast
        y = np.random.randn(100, 10)
        x = np.random.randn(100, 5)
        trial = {'y': y, 'x': x}
        fast_trials = fast.cut_trial(trial, 10)
        for each in fast_trials:
            self.assertTrue(each['y'].shape == (10, 10))
