import os
from time import time

import numba as nb
import numpy as np
# noinspection PyPackageRequirements
import pytest
from numpy.testing import assert_allclose
# noinspection PyPackageRequirements
from scipy.stats import wilcoxon as scp_wilcoxon

from permutations_stats.tests import wilcoxon


# noinspection DuplicatedCode
class Test:

    def setup_class(self):
        about = {}
        with open("permutations_stats/__about__.py") as fp:
            exec(fp.read(), about)
        self.version = about['__version__']

        numba_disabled = os.environ.get("NUMBA_DISABLE_JIT", False)
        self.ps_label = "PSN" if numba_disabled else "PSB"

    def run(self, test_id, n_tests, n_x, n_y):
        print()
        rng = np.random.default_rng(seed=n_tests*100 + n_x*10 + n_y)
        x = rng.random((n_tests, n_x))
        y = rng.random((n_tests, n_y)) + 0.02

        time_ps = 0
        time_scp = 0

        for i in range(n_tests):
            x_i = x[i, :]
            y_i = y[i, :]
            t0 = time()
            w_input = np.stack((x_i, y_i), axis=1)
            ans_ps = wilcoxon.test(w_input, return_w=2)
            t1 = time()

            t2 = time()
            ans_scp = scp_wilcoxon(x_i, y_i, alternative="two-sided")[0]
            t3 = time()

            time_ps += t1 - t0
            time_scp += t3 - t2
            assert_allclose(ans_ps, ans_scp)

        print(f"{n_tests} tests with {n_x} and {n_y} data points - "
              f"Permutations-stats: {time_ps:.3f}s, Scipy: {time_scp:.3f}s, "
              f"diff: {time_scp - time_ps:.3f}s")

        with open("speeds.csv", 'a') as speeds_file:
            speeds_file.write(f"{self.version};WX;{test_id};"
                              f"{self.ps_label};{time_ps};SCP;{time_scp};"
                              f"{time_scp - time_ps}\n")

    def test_100_5_5groups(self):
        n_x = 5
        n_y = 5
        n_tests = 200

        self.run(1, n_tests, n_x, n_y)

    def test_100_3_3groups(self):
        n_x = 3
        n_y = 3
        n_tests = 200

        self.run(2, n_tests, n_x, n_y)

    def test_20_10_10groups(self):
        n_x = 10
        n_y = 10
        n_tests = 200

        self.run(3, n_tests, n_x, n_y)

    def test_10000_14_14groups(self):
        n_x = 14
        n_y = 14
        n_tests = 10_000

        self.run(4, n_tests, n_x, n_y)

    def test_20000_15_15groups(self):
        n_x = 15
        n_y = 15
        n_tests = 20_000

        self.run(5, n_tests, n_x, n_y)

    def test_30000_18_18groups(self):
        n_x = 18
        n_y = 18
        n_tests = 30_000

        self.run(6, n_tests, n_x, n_y)

    def test_input_not1ds(self):
        rng = np.random.default_rng()
        x = rng.random((5, 5))
        y = rng.random((5, 5))

        with pytest.raises(TypeError):
            wilcoxon.test(np.stack((x, y)), y)

    # noinspection PyTypeChecker
    def test_input_not_numeric_array(self):

        rng = np.random.default_rng()
        x = rng.random((5, 5))

        with pytest.raises((TypeError, nb.TypingError)):
            wilcoxon.test("this", x)

        with pytest.raises((TypeError, nb.TypingError)):
            wilcoxon.test(x, "this")
