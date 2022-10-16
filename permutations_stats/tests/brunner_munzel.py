import math

import numba as nb
import numpy as np

from permutations_stats.utils import rank_1d


@nb.njit()
def test(x: np.ndarray, y: np.ndarray):
    # noinspection PyBroadException
    try:
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise TypeError("Input should be 1D arrays.")

    except:  # noqa: E722
        raise TypeError(
            "Please provide a numeric-valued 1D numpy arrays for x and y.")

    return _test(x, y)[0]


@nb.njit()
def _test(x: np.ndarray, y: np.ndarray):
    # noinspection PyBroadException
    try:
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise TypeError("Input should be 1D arrays.")

    except:  # noqa: E722
        raise TypeError(
            "Please provide a numeric-valued 1D numpy arrays for x and y.")

    n_x = len(x)
    n_y = len(y)
    n_tot = n_x + n_y
    tn = _test_faster(x, y, (n_x, n_y, n_tot))

    if not np.isinf(tn):
        tn = tn * math.sqrt(n_x * n_y / n_tot)

    return tn, (n_x, n_y, n_tot)


@nb.njit()
def _test_faster(x: np.ndarray, y: np.ndarray, args):
    n_x, n_y, n_tot = args

    o_r_x = rank_1d(x)
    o_r_y = rank_1d(y)

    n_ranks = rank_1d(np.concatenate((x, y)))

    n_r_x = n_ranks[:n_x]
    n_r_y = n_ranks[n_x:]

    avg_ranks_x = np.mean(n_ranks[:n_x])
    avg_ranks_y = np.mean(n_ranks[n_x:])

    p_hat = avg_ranks_y - avg_ranks_x

    r_diff_x = n_r_x - o_r_x
    r_diff_y = n_r_y - o_r_y

    s2x = (np.sum(np.power((r_diff_x - avg_ranks_x + (n_x + 1) / 2), 2))
           / (n_x - 1))
    s2y = (np.sum(np.power((r_diff_y - avg_ranks_y + (n_y + 1) / 2), 2))
           / (n_y - 1))

    v_n = n_tot * (s2x / n_y + s2y / n_x)

    if v_n == 0:
        return np.sign(p_hat) * np.inf

    tn = p_hat / math.sqrt(v_n)

    return tn


@nb.njit()
def abs_test(x: np.ndarray, y: np.ndarray):
    tn, args = _test(x, y)
    return np.abs(tn), args


@nb.njit()
def invert_test(x: np.ndarray, y: np.ndarray):
    return _test(y, x)


@nb.njit()
def abs_test_faster(x: np.ndarray, y: np.ndarray, args):
    return np.abs(_test_faster(x, y, args))


@nb.njit()
def invert_test_faster(x: np.ndarray, y: np.ndarray, args):
    return _test_faster(y, x, args)
