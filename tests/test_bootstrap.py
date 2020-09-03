import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from permutations_stats.bootstrap import bootstrap_mean, bootstrap_median, \
                                         bootstrap_std


def test_confidence_interval():
    rng = np.random.default_rng(seed=2020)
    sample_size = 100_000
    x = rng.normal(size=sample_size)
    mean, lower, upper = bootstrap_mean(x, 95, n_iter=10000, seed=2020)

    mean_e = np.mean(x)
    lower_e = mean_e - 1.96 * np.std(x) / sample_size ** .5
    upper_e = mean_e + 1.96 * np.std(x) / sample_size ** .5

    assert_almost_equal(mean, mean_e)
    assert_almost_equal(lower, lower_e, decimal=4)
    assert_almost_equal(upper, upper_e, decimal=4)


def test_mean():
    a = np.arange(10)
    assert_almost_equal(bootstrap_mean(a, 95, 100)[0], 4.5)


def test_mean_with_seed():
    a = np.arange(10)
    assert_allclose(bootstrap_mean(a, 95, 100, seed=20),
                    bootstrap_mean(a, 95, 100, seed=20))


def test_median():
    a = np.arange(10)
    assert_almost_equal(bootstrap_median(a, 95, 100)[0], 4.5)


def test_median_with_seed():
    a = np.arange(10)
    assert_allclose(bootstrap_median(a, 95, 100, seed=20),
                    bootstrap_median(a, 95, 100, seed=20))


def test_std():
    a = np.arange(10)
    assert_almost_equal(bootstrap_median(a, 95, 100)[0], 4.5)


def test_std_with_seed():
    a = np.arange(10)
    assert_allclose(bootstrap_std(a, 95, 100, seed=20),
                    bootstrap_std(a, 95, 100, seed=20))
