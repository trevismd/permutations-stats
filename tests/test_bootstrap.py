import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
# noinspection PyProtectedMember
from permutations_stats.bootstrap import bootstrap_mean, bootstrap_median, \
    bootstrap_std, _bootstrap_median_pll, _bootstrap_std_pll, \
    _bootstrap_mean_pll


class Test:

    @classmethod
    def setup_class(cls):
        cls.rng = np.random.default_rng(seed=2020)
        cls.sample_size = 100_000
        cls.x = cls.rng.normal(size=cls.sample_size)
        cls.n_iter = 10_000
        cls.a = np.arange(10)
        cls.b = np.arange(1000)

    def test_confidence_interval_mean(self):

        mean, lower, upper = bootstrap_mean(
            self.x, 95, n_iter=self.n_iter, seed=2020, verbose=True)

        mean_p, lower_p, upper_p = _bootstrap_mean_pll(
            self.x, 95, n_iter=self.n_iter)

        mean_e = np.mean(self.x)
        lower_e = mean_e - 1.96 * np.std(self.x) / self.sample_size ** .5
        upper_e = mean_e + 1.96 * np.std(self.x) / self.sample_size ** .5

        assert_almost_equal(mean, mean_e)

        assert_allclose(lower, lower_e, atol=1e-3)
        assert_allclose(upper, upper_e, atol=1e-3)

        assert_allclose(lower_p, lower, atol=1e-3)
        assert_allclose(upper_p, upper, atol=1e-3)

    def test_confidence_interval_median_pll(self):
        median, lower, upper = bootstrap_median(
            self.x, 95, n_iter=self.n_iter, seed=2020, verbose=True)
        median_p, lower_p, upper_p = _bootstrap_median_pll(
            self.x, 95, n_iter=self.n_iter)

        median_e = np.median(self.x)

        assert_allclose(median, median_e, atol=2e-4)
        assert_allclose(median_p, median_e, atol=2e-4)

        assert_allclose(lower_p, lower, atol=2e-4)
        assert_allclose(upper_p, upper, atol=2e-4)

    def test_confidence_interval_std_pll(self):
        std, lower, upper = bootstrap_std(
            self.x, 95, n_iter=self.n_iter, seed=2020, verbose=True)
        std_p, lower_p, upper_p = _bootstrap_std_pll(
            self.x, 95, n_iter=self.n_iter)

        std_e = np.std(self.x)

        assert_allclose(std, std_e, atol=3e-4)
        assert_allclose(std_p, std_e, atol=3e-4)

        assert_allclose(lower_p, lower, atol=3e-4)
        assert_allclose(upper_p, upper, atol=3e-4)

    def test_mean(self):
        assert_almost_equal(bootstrap_mean(self.a, 95, 100, verbose=True)[0],
                            4.5)

    def test_mean_numba(self):
        assert_almost_equal(bootstrap_mean(self.b, 95, 100, verbose=True)[0],
                            499.5)

    def test_mean_with_seed(self):
        assert_allclose(bootstrap_mean(self.a, 95, 100, seed=20, verbose=True),
                        bootstrap_mean(self.a, 95, 100, seed=20, verbose=True))

    def test_median(self):
        assert_almost_equal(bootstrap_median(self.a, 95, 100, verbose=True)[0],
                            4.5)

    def test_median_numba(self):
        assert_almost_equal(bootstrap_median(self.b, 95, 100, verbose=True)[0],
                            499.5)

    def test_median_with_seed(self):
        assert_allclose(bootstrap_median(self.a, 95, 100, seed=20, verbose=True),
                        bootstrap_median(self.a, 95, 100, seed=20, verbose=True))

    def test_std(self):
        assert_allclose(bootstrap_std(self.a, 95, 100, verbose=True)[0],
                        np.std(self.a))

    def test_std_numba(self):
        assert_allclose(bootstrap_std(self.b, 95, 1000, verbose=True)[0],
                        np.std(self.b))

    def test_std_with_seed(self):
        assert_allclose(bootstrap_std(self.a, 95, 100, seed=20, verbose=True),
                        bootstrap_std(self.a, 95, 100, seed=20, verbose=True))
