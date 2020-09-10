import numba as nb
import numpy as np

from permutations_stats.utils import rank_1d


@nb.njit()
def test(x: np.ndarray, return_w=0):
    """
    returns: W if return_w is 0, W+ if 1 (R), min(W+, W-) if 2 (scipy)
    """
    # noinspection PyBroadException
    try:
        if len(x.shape) != 2 or x.shape[1] != 2:
            print(x.shape)
            raise TypeError("Input should be a (n, 2) array.")

    except:
        raise TypeError("Please provide a numeric-valued 2D numpy array for x.")

    diffs = x[:, 1] - x[:, 0]

    signs = np.sign(diffs)
    diffs = np.abs(diffs)

    ranks = rank_1d(diffs)
    ranks = np.where(diffs > 1e-7, ranks, 0)

    if return_w == 1:
        ranks = np.where(signs > 0, ranks, 0)

    rank_sum = signs.T @ ranks

    if return_w == 2:

        if rank_sum > 0:
            ranks = np.where(signs < 0, ranks, 0)
        else:
            ranks = np.where(signs > 0, ranks, 0)
        rank_sum = np.sum(ranks)

    return rank_sum
