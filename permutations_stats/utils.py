import numba as nb
import numpy as np
from enum import Enum


class Alternative(Enum):
    TWO_SIDED = 1
    GREATER = 2
    LESS = 3


class Method(Enum):
    exact = 1
    simulation = 2


@nb.vectorize([nb.b1(nb.float64, nb.float64)])
def is_close(x, y):
    # Similar to numpy isclose
    x_fin = np.isfinite(x)
    y_fin = np.isfinite(y)

    if x_fin and y_fin:
        return np.abs(x - y) < (1.e-8 + 1.e-5 * abs(y))
    else:
        return x_fin == y_fin


@nb.njit()
def rank_2d(array):
    """ Ranks all values of a 2d array, averaging ties.
    :param array: 2d numpy array
    :returns: array whose values are replaced by their rank
    """
    n_array = array.flatten()
    ranked = rank_1d(n_array)

    return ranked.reshape(array.shape)


@nb.njit()
def rank_2d_by_col(array):
    """ Ranks all columns of a 2d array, averaging ties.
    :param array: 2d numpy array
    :returns: array whose values are replaced by their rank by column
    """
    n_arrays = array.shape[1]
    result = np.empty_like(array)
    for array_idx in range(n_arrays):
        result[:, array_idx] = rank_1d(array[:, array_idx])
    return result


@nb.njit()
def rank_2d_by_row(array):
    """ Ranks rows of a 2d array, averaging ties.
    :param array: 2d numpy array
    :returns: array whose values are replaced by their rank by row
    """
    return rank_2d_by_col(array.T).T


@nb.njit()
def rank_1d(array):
    # Thanks Sven Mamach - https://stackoverflow.com/a/5284703
    # and Martin F Thomsen - https://stackoverflow.com/a/20455974
    """ Rank a 1d array, with averages of ties.
    :param array: 1d numpy array
    :returns: array whose values are replaced by their rank
    """
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


def simpler_ranks_1d(array):
    # Thanks Sven Mamach - https://stackoverflow.com/a/5284703
    # and Martin F Thomsen - https://stackoverflow.com/a/20455974
    """ Rank a 1d array, with averages of ties.
    :param array: 1d numpy array
    :returns: array whose values are replaced by their rank
    """
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    to_check = np.ones_like(array)
    for i in range(len(array)):
        if not to_check[i]:
            continue
        same = array == array[i]
        same_count = np.sum(same)

        if same_count > 1:
            ranks_min = np.min(ranks[same])
            ranks[same] = ranks_min

            greater = array > array[i]
            ranks[greater] -= same_count - 1
            to_check[same] = False

    return ranks


def simpler_ranks(array):
    new_array = np.empty_like(array)
    for row_idx, row in enumerate(array):
        new_array[row_idx] = simpler_ranks_1d(row)
    return new_array.astype(np.int64)


# This section is adapted from @joelrich
# https://github.com/numba/numba/issues/1269#issuecomment-472574352
@nb.njit
def np_apply_along_axis(func1d, axis, arr):

    if arr.ndim != 2:
        raise NotImplementedError("This function applies to 2d arrays")

    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])

    elif axis == 1:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])

    else:
        raise ValueError("Axis must be 0 or 1")

    return result


@nb.njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@nb.njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


@nb.njit
def np_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)
