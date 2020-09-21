import numpy as np
# noinspection PyPackageRequirements
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose

import permutations_stats.permutations as pm
import permutations_stats.utils as pmu


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


def test_greater_alternative():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)

    assert_almost_equal(pm.permutation_test(x, y, alternative="g")[1],
                        pm.permutation_test(y, x, alternative="l")[1])


def test_correct_func():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)

    with pytest.raises(TypeError):
        pm.permutation_test(x, y, stat_func="that one")


def test_correct_test():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)

    with pytest.raises(ValueError):
        pm.permutation_test(x, y, test="this one")


def test_correct_test_param_type():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)

    with pytest.raises(TypeError):
        pm.permutation_test(x, y, test=np.mean)


def test_permutation_test_not_array_like():
    with pytest.raises(TypeError):
        pm.permutation_test("avcd", [1, 2, 3])


def test_permutation_test_array_like():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)
    assert_almost_equal(
        pm.permutation_test(x,
                            y,
                            method="approximate", n_iter=100)[1],
        pm.permutation_test(x.tolist(),
                            y.tolist(),
                            method="approximate", n_iter=100)[1])


def test_valid_method():
    with pytest.raises(ValueError):
        pm._check_and_get("friedman", "two-sided", None, "that method")


def test_override_method():
    assert pm._check_method(100, 100, pmu.Method.simulation, False) == pmu.Method.exact


def test_force_no_override_method():
    assert pm._check_method(1000, 100, pmu.Method.simulation, True) == pmu.Method.simulation


def test_permutations_w_seed():
    rng = np.random.default_rng(seed=100)
    x = rng.random(5)
    y = rng.random(5)

    assert_allclose(
        pm.permutation_test(x, y, method="approximate", n_iter=100, seed=100)[:2],
        pm.permutation_test(x, y, method="approximate", n_iter=100, seed=100)[:2])
