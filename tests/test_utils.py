import numpy as np
from numpy.testing import assert_allclose
from permutations_stats.utils import rank_1d, rank_2d, rank_2d_by_col, \
                                     rank_2d_by_row
from permutations_stats.utils import is_close


def test_simple_rank_1d():
    r1 = np.array([1, 0, 8, 6, 3, 2, 33, 9])
    assert_allclose(rank_1d(r1),
                    np.array([2, 1, 6, 5, 4, 3, 8, 7]))


def test_simple_rank_2d():
    r1 = np.array([[1, 0, 8, 6], [3, 2, 33, 9]])
    assert_allclose(rank_2d(r1),
                    np.array([[2, 1, 6, 5], [4, 3, 8, 7]]))


def test_rank_2d_by_col1():
    r1 = np.array([[1, 0, 8, 6], [3, 2, 33, 9]])
    assert_allclose(rank_2d_by_col(r1),
                    np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))


def test_rank_2d_by_col2():
    r1 = np.array([[1, 0, 8, 6], [3, 2, 33, 9]]).T
    assert_allclose(rank_2d_by_col(r1),
                    np.array([[2, 1, 4, 3], [2, 1, 4, 3]]).T)


def test_rank_2d_by_row1():
    r1 = np.array([[1, 0, 8, 6], [3, 2, 33, 9]])
    assert_allclose(rank_2d_by_row(r1),
                    np.array([[2, 1, 4, 3], [2, 1, 4, 3]]))


def test_rank_2d_by_row2():
    r1 = np.array([[1, 0, 8, 6], [3, 2, 33, 9]]).T
    assert_allclose(rank_2d_by_row(r1),
                    np.array([[1, 1, 1, 1], [2, 2, 2, 2]]).T)


def test_is_close():
    a = 2.000000000000001
    b = 3. - 1.
    assert a != b
    assert is_close(a, b)
