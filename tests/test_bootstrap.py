import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
# noinspection PyProtectedMember
from permutations_stats.bootstrap import bootstrap_mean, bootstrap_median, \
    bootstrap_std, _bootstrap_median_pll, _bootstrap_std_pll, \
    _bootstrap_mean_pll


class Test:

    def test_confidence_interval_mean(self):
        rng = np.random.default_rng(seed=2020)
        sample_size = 100_000
        x = rng.normal(size=sample_size)
        mean, lower, upper = bootstrap_mean(x, 95, n_iter=10_000, seed=2020,
                                            verbose=True)

        mean_p, lower_p, upper_p = _bootstrap_mean_pll(x, 95, n_iter=10_000)

        mean_e = np.mean(x)
        lower_e = mean_e - 1.96 * np.std(x) / sample_size ** .5
        upper_e = mean_e + 1.96 * np.std(x) / sample_size ** .5

        assert_almost_equal(mean, mean_e)

        assert_allclose(lower, lower_e, atol=2e-4)
        assert_allclose(upper, upper_e, atol=2e-4)

        assert_allclose(lower_p, lower, atol=2e-4)
        assert_allclose(upper_p, upper, atol=2e-4)

    def test_confidence_interval_median_pll(self):
        rng = np.random.default_rng(seed=2020)
        sample_size = 100_000
        n_iter=10_000
        x = rng.normal(size=sample_size)
        median, lower, upper = bootstrap_median(x, 95, n_iter=n_iter, seed=2020,
                                                verbose=True)
        median_p, lower_p, upper_p = _bootstrap_median_pll(x, 95, n_iter)

        median_e = np.median(x)

        assert_allclose(median, median_e, atol=2e-4)
        assert_allclose(median_p, median_e, atol=2e-4)

        assert_allclose(lower_p, lower, atol=2e-4)
        assert_allclose(upper_p, upper, atol=2e-4)

    def test_confidence_interval_std_pll(self):
        rng = np.random.default_rng(seed=2020)
        sample_size = 100_000
        n_iter=10_000
        x = rng.normal(size=sample_size)
        std, lower, upper = bootstrap_std(x, 95, n_iter=n_iter, seed=2020,
                                          verbose=True)
        std_p, lower_p, upper_p = _bootstrap_std_pll(x, 95, n_iter)

        std_e = np.std(x)

        assert_allclose(std, std_e, atol=2e-4)
        assert_allclose(std_p, std_e, atol=2e-4)

        assert_allclose(lower_p, lower, atol=2e-4)
        assert_allclose(upper_p, upper, atol=2e-4)

    def test_mean(self):
        a = np.arange(10)
        assert_almost_equal(bootstrap_mean(a, 95, 100, verbose=True)[0], 4.5)

    def test_mean_numba(self):
        a = np.arange(1000)
        assert_almost_equal(bootstrap_mean(a, 95, 100, verbose=True)[0], 499.5)

    def test_mean_numba(self):
        a = np.arange(1000)
        assert_almost_equal(bootstrap_mean(a, 95, 100, verbose=True)[0], 499.5)

    def test_mean_with_seed(self):
        a = np.arange(10)
        assert_allclose(bootstrap_mean(a, 95, 100, seed=20, verbose=True),
                        bootstrap_mean(a, 95, 100, seed=20, verbose=True))

    def test_median(self):
        a = np.arange(10)
        assert_almost_equal(bootstrap_median(a, 95, 100, verbose=True)[0], 4.5)

    def test_median_numba(self):
        a = np.arange(1000)
        assert_almost_equal(bootstrap_median(a, 95, 100, verbose=True)[0], 499.5)

    def test_median_with_seed(self):
        a = np.arange(10)
        assert_allclose(bootstrap_median(a, 95, 100, seed=20, verbose=True),
                        bootstrap_median(a, 95, 100, seed=20, verbose=True))

    def test_std(self):
        a = np.arange(10)
        assert_allclose(bootstrap_std(a, 95, 100, verbose=True)[0], np.std(a))

    def test_std_numba(self):
        a = np.arange(1000)
        assert_allclose(bootstrap_std(a, 95, 1000, verbose=True)[0], np.std(a))

    def test_std_with_seed(self):
        a = np.arange(10)
        assert_allclose(bootstrap_std(a, 95, 100, seed=20, verbose=True),
                        bootstrap_std(a, 95, 100, seed=20, verbose=True))
