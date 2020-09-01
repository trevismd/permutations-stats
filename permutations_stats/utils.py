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
        result[:, array_idx] = rank_1d(array[:,array_idx])
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


# This section is adapted from @joelrich
# https://github.com/numba/numba/issues/1269#issuecomment-472574352
@nb.njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result


@nb.njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@nb.njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)
