import numba as nb
import numpy as np

from permutations_stats.utils import rank_1d


@nb.njit()
def both_u(x: np.ndarray, y: np.ndarray):
    # noinspection PyBroadException
    try:
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise TypeError("Input should be 1D arrays.")

    except:
        raise TypeError("Please provide a numeric-valued 1D numpy arrays for x and y.")

    n_x = len(x)
    n_y = len(y)

    ranks = rank_1d(np.concatenate((x, y)))
    rank_sum_x = np.sum(ranks[:n_x])

    u = rank_sum_x - n_x * (n_x + 1) / 2.0
    us = (u, n_x * n_y - u)

    return us


@nb.njit()
def test(x: np.ndarray, y: np.ndarray):

    return both_u(x, y)[0]


@nb.njit()
def min_u(x: np.ndarray, y: np.ndarray):

    return min(both_u(x, y))


@nb.njit()
def max_u(x: np.ndarray, y: np.ndarray):

    return max(both_u(x, y))
