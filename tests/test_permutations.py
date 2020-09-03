import permutations_stats.permutations as pm
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest


def test_approximate_override():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)
    assert_equal(
        pm.permutation_test(x, y, method="approximate", n_iter=1000)[2],
        252)


def test_approximate_n_iter():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)
    assert_equal(
        pm.permutation_test(x, y, method="approximate", n_iter=100)[2],
        101)


def test_approximate_p_val():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)
    assert_almost_equal(
        pm.permutation_test(x, y, method="approximate", n_iter=100, seed=100)[1],
        0.96039604)


def test_correct_alternative():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)

    with pytest.raises(ValueError):
        pm.permutation_test(x, y, alternative="this")


def test_counts():
    array = np.array([1, 2, 3, 4])
    res_greater, res_smaller, res_equal = pm.counts(array, 2)
    assert_equal(res_greater, 2)
    assert_equal(res_equal, 1)
    assert_equal(res_smaller, 1)
