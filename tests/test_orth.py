from unittest import TestCase

import numpy as np

from vlgp.math import orth


class TestOrth(TestCase):
    def test_orth(self):
        np.random.seed(0)
        n = 500
        p = 200
        q = 100
        x = np.random.randn(n, q)
        a = np.random.rand(q, p)
        x_orth, a_orth = orth(x, a)
        self.assertTrue(np.allclose(x @ a, x_orth @ a_orth))
