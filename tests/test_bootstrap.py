import numpy as np
from numpy.testing import assert_almost_equal

from permutations_stats.bootstrap import bootstrap_mean, bootstrap_median


def test_mean():
    a = np.arange(10)
    assert_almost_equal(bootstrap_mean(a, 0.05, 100)[0], 4.5)


def test_median():
    a = np.arange(10)
    assert_almost_equal(bootstrap_median(a, 0.05, 100)[0], 4.5)