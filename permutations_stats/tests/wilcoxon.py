"""
This implementation of Wilcoxon's signed rank test takes an input with shape
(N, 2). When using a "greater" alternative, as in other packages, it means that
the first dimension values are greater than the second.
"""

import numba as nb
import numpy as np

from permutations_stats.utils import rank_1d


@nb.njit()
def test(x: np.ndarray, return_w=0.):
    """

    returns: W if return_w is 0, W+ if 1 (R), min(W+, W-) if 2 (scipy)
    """
    # noinspection PyBroadException
    try:
        if len(x.shape) != 2 or x.shape[1] != 2:
            print(x.shape)
            raise TypeError("Input should be a (n, 2) array.")

    except:  # noqa: E722
        raise TypeError(
            "Please provide a numeric-valued 2D numpy array for x.")

    data = _test(x, return_w)

    return data[0]


@nb.njit()
def _test(x: np.ndarray, return_w=0.):
    """

    returns: W if return_w is 0, W+ if 1 (R), min(W+, W-) if 2 (scipy)
    """
    # noinspection PyBroadException
    try:
        if len(x.shape) != 2 or x.shape[1] != 2:
            print(x.shape)
            raise TypeError("Input should be a (n, 2) array.")

    except:  # noqa: E722
        raise TypeError(
            "Please provide a numeric-valued 2D numpy array for x.")

    rank_sum = _test_faster(x, np.array([return_w]))

    return rank_sum, return_w


@nb.njit()
def _test_faster(x: np.ndarray, return_w):
    diffs = x[:, 0] - x[:, 1]

    signs = np.sign(diffs)
    diffs = np.abs(diffs)

    ranks = rank_1d(diffs)
    ranks = np.where(diffs > 1e-7, ranks, 0)

    if return_w == 0:
        rank_sum = signs.T @ ranks

    elif return_w == 1:
        ranks = np.where(signs > 0, ranks, 0)

        rank_sum = signs.T @ ranks

    else:  # return_w == 2
        rank_sum = signs.T @ ranks

        if rank_sum > 0:
            ranks = np.where(signs < 0, ranks, 0)
        else:
            ranks = np.where(signs > 0, ranks, 0)

        rank_sum = np.sum(ranks)

    return rank_sum


@nb.njit()
def abs_test(x: np.ndarray):
    data = _test(x)
    return np.abs(data[0]), data[1]


@nb.njit()
def abs_test_faster(x: np.ndarray, args):
    return np.abs(_test_faster(x, args))
