import numba as nb
import numpy as np
# noinspection PyPackageRequirements
import pytest

from permutations_stats.tests import friedman

# noinspection DuplicatedCode
fd_array = np.array(
    [[103.3, 68.42, 89.53, 77.94, 100.0, 108.2, 184.9],
     [42.19, 44.31, 60.91, 73.90, 43.87, 61.74, 102.3],
     [71.27, 81.88, 100.71, 86.52, 100.3, 90.75, 100.6],
     [37.59, 60.05, 56.97, 60.79, 71.82, 83.04, 117.1],
     [58.31, 52.73, 96.04, 60.42, 104.33, 89.78, 85.77],
     [46.27, 82.18, 129.8, 181.0, 172.33, 164.8, 246.8],
     [19.00, 23.07, 38.70, 45.81, 59.03, 50.69, 55.18],
     [8.31, 8.43, 9.16, 14.28, 10.63, 15.84, 12.50],
     [20.15, 33.48, 60.08, 69.35, 114.34, 45.28, 101.6],
     [3.16, 4.12, 12.73, 18.95, 8.89, 41.52, 66.33],
     [4.12, 18.87, 8.54, 12.92, 25.30, 19.85, 16.76],
     [7.68, 11.18, 10.44, 10.95, 10.54, 13.96, 14.39],
     [5.29, 10.91, 11.22, 25.26, 42.25, 48.80, 69.38],
     [6.00, 5.57, 22.23, 2.45, 6.24, 1.00, 4.00]])


def test_statistic():
    np.testing.assert_almost_equal(friedman.test(fd_array),
                                   40.1020408)


def test_ties_correction(capsys):
    friedman.test(np.array([[1, 1, 3],
                            [1, 2, 3]]),
                            verbose=True)
    captured = capsys.readouterr()
    assert captured.out == f"{friedman.CONOVER_APPLIED}\n"


def test_input_not2d():
    # noinspection PyTypeChecker
    with pytest.raises((NotImplementedError, TypeError, nb.TypingError)):
        friedman.test(np.array([1, 2, 3]))


# noinspection PyTypeChecker
def test_input_not_numeric_array():
    with pytest.raises((TypeError, nb.TypingError)):
        friedman.test("this")
