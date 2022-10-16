import numba as nb
import numpy as np

from permutations_stats.utils import rank_1d


@nb.njit()
def both_u(x: np.ndarray, y: np.ndarray):
    # noinspection PyBroadException
    try:
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise TypeError("Input should be 1D arrays.")

    except:  # noqa: E722
        raise TypeError(
            "Please provide a numeric-valued 1D numpy arrays for x and y.")

    n_x = len(x)
    n_y = len(y)
    us = both_u_faster(x, y, (n_x, n_y))

    return us, (n_x, n_y)


@nb.njit()
def both_u_faster(x: np.ndarray, y: np.ndarray, args):
    n_x, n_y = args
    ranks = rank_1d(np.concatenate((x, y)))
    rank_sum_x = np.sum(ranks[:n_x])

    u = rank_sum_x - n_x * (n_x + 1) / 2.0
    us = (u, n_x * n_y - u)

    return us


@nb.njit()
def test(x: np.ndarray, y: np.ndarray):
    data = both_u(x, y)
    return data[0][0], data[1]


@nb.njit()
def test_faster(x: np.ndarray, y: np.ndarray, args):
    return both_u_faster(x, y, args)[0]


@nb.njit()
def min_u(x: np.ndarray, y: np.ndarray):
    data = both_u(x, y)
    return min(data[0]), data[1]


@nb.njit()
def min_u_faster(x: np.ndarray, y: np.ndarray, args):
    return min(both_u_faster(x, y, args))


@nb.njit()
def max_u(x: np.ndarray, y: np.ndarray):
    data = both_u(x, y)
    return max(data[0], data[1])


@nb.njit()
def max_u_faster(x: np.ndarray, y: np.ndarray, args):
    return max(both_u_faster(x, y, args))
