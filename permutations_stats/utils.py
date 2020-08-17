import numba as nb
import numpy as np


@nb.vectorize([nb.b1(nb.float64, nb.float64)])
def is_close(x, y):
    # Similar to numpy isclose
    x_fin = np.isfinite(x)
    y_fin = np.isfinite(y)

    if x_fin and y_fin:
        return np.abs((x - y)) < (1.e-8 + (1.e-5 * abs(y)))
    else:
        return x_fin == y_fin


# @nb.njit(nb.float64[:](nb.float64[:]))
@nb.njit()
def rank_array(array):
    # Thanks Sven Mamach - https://stackoverflow.com/a/5284703
    # and Martin F Thomsen - https://stackoverflow.com/a/20455974
    temp = array.argsort()
    ranks = np.empty_like(temp, dtype=np.float64)
    ranks[temp] = np.arange(len(array), dtype=np.float64)
    to_check = np.ones_like(array)

    for i in range(len(array)):
        if not to_check[i]:
            continue
        same = array == array[i]
        same_count = np.sum(same)

        if same_count > 1:
            ranks_sum = np.sum(ranks[same])
            ranks[same] = float(ranks_sum) / same_count
        to_check[same] = False

    return ranks + 1
