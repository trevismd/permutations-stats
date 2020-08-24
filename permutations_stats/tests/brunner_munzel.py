import math

import numba as nb
import numpy as np

from permutations_stats.utils import rank_array


@nb.njit()
def test(x, y):
    n_x = len(x)
    n_y = len(y)
    n_tot = n_x + n_y

    o_r_x = rank_array(x)
    o_r_y = rank_array(y)

    n_ranks = rank_array(np.concatenate((x, y)))

    n_r_x = n_ranks[:n_x]
    n_r_y = n_ranks[n_x:]

    avg_ranks_x = np.mean(n_ranks[:n_x])
    avg_ranks_y = np.mean(n_ranks[n_x:])

    p_hat = avg_ranks_y - avg_ranks_x

    r_diff_x = n_r_x - o_r_x
    r_diff_y = n_r_y - o_r_y

    s2x = np.sum(np.power((r_diff_x - avg_ranks_x + (n_x + 1) / 2), 2)) / (n_x - 1)
    s2y = np.sum(np.power((r_diff_y - avg_ranks_y + (n_y + 1) / 2), 2)) / (n_y - 1)

    v_n = n_tot * s2x / n_y + n_tot * s2y / n_x

    if v_n == 0:
        return np.sign(p_hat) * np.inf

    tn = p_hat / math.sqrt(v_n) * math.sqrt(n_x * n_y / n_tot)

    return tn
