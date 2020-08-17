import warnings

import numpy as np
from scipy.stats import brunnermunzel
from time import time
from permutations_stats.tests import brunner_munzel


def run(n_tests, n_x, n_y):
    print()
    rng = np.random.default_rng(seed=n_tests*100 + n_x*10 + n_y)
    x = rng.random((n_tests, n_x))
    y = rng.random((n_tests, n_y)) + 0.02

    ans_ps = np.empty((n_tests, 1))
    ans_scp = np.empty((n_tests, 1))

    time_ps = 0
    time_scp = 0

    for i in range(n_tests):
        x_i = x[i, :]
        y_i = y[i, :]
        t0 = time()
        ans_ps = brunner_munzel.test(x_i, y_i)
        t1 = time()

        with warnings.catch_warnings():  # From scipy implementation
            warnings.simplefilter("ignore")
            t2 = time()
            ans_scp = brunnermunzel(x_i, y_i)[0]
            t3 = time()

        time_ps += t1 - t0
        time_scp += t3 - t2

    print(f"{n_tests} tests with {n_x} and {n_y} data points - "
          f"Permutations-stats: {time_ps:.3f} s, Scipy: {time_scp:.3f}s, "
          f"diff: {time_scp - time_ps:.3f}")

    return np.allclose(ans_ps, ans_scp)


def test_100_5_4groups():
    n_x = 5
    n_y = 4
    n_tests = 200

    assert run(n_tests, n_x, n_y)


def test_100_3_3groups():
    n_x = 3
    n_y = 3
    n_tests = 200

    assert run(n_tests, n_x, n_y)


def test_20_10_12groups():
    n_x = 10
    n_y = 12
    n_tests = 200

    assert run(n_tests, n_x, n_y)


def test_10000_18_10groups():
    n_x = 18
    n_y = 10
    n_tests = 10_000

    assert run(n_tests, n_x, n_y)


def test_20000_18_15groups():
    n_x = 18
    n_y = 15
    n_tests = 20_000

    assert run(n_tests, n_x, n_y)


def test_30000_18_19groups():
    n_x = 18
    n_y = 19
    n_tests = 30_000

    assert run(n_tests, n_x, n_y)
