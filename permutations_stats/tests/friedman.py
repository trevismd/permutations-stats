import numba as nb
import numpy as np

import permutations_stats.utils as pmu

CONOVER_APPLIED = "Conover correction for ties applied"


@nb.njit()
def transform(array):
    return pmu.rank_2d_by_row(array)


@nb.njit()
def test(array: np.ndarray, t=1, verbose=False):
    """
    Subjects are array first dimension.
    Treatments are the array second condition.
    In case of ties Conover's statistic is calculated and T2 is returned
    instead of T1.
    :params array: Data in shape (n_subjects, n_treatments)
    :return Friedman statistic
    """
    # noinspection PyBroadException
    try:
        if len(array.shape) != 2:
            raise NotImplementedError("Input should be a 2D array.")

    except:  # noqa: E722
        raise TypeError(
            "Please provide a numeric-valued 2D numpy arrays for x.")

    return _test(array, t, verbose)[0]


@nb.njit()
def test_t2(array: np.ndarray, verbose=False):
    return _test(array, 2, verbose)[0]


@nb.njit()
def _test_t2(array: np.ndarray, verbose=False):
    return _test(array, 2, verbose)


# @nb.njit(['float64(float64[:, :])',
#           'float64(float32[:, :])',
#           'float64(int32[:, :])',
#           'float64(int64[:, :])'])
@nb.njit()
def _test(array: np.ndarray, t=1, verbose=False):
    """
    Subjects are array first dimension.
    Treatments are the array second condition.
    In case of ties Conover's statistic is calculated. Pass t=2 to return t2
    :param array: Data in shape (n_subjects, n_treatments)
    :param t: pass t=2 to return t2 in case of ties
    :return Friedman statistic
    """
    # noinspection PyBroadException
    try:
        if len(array.shape) != 2:
            raise NotImplementedError("Input should be a 2D array.")

    except:  # noqa: E722
        raise TypeError(
            "Please provide a numeric-valued 2D numpy arrays for x.")

    n_subjects = array.shape[0]
    n_treatments = array.shape[1]

    ranked_data = pmu.rank_2d_by_row(array)
    treatment_sums = pmu.np_sum(ranked_data, axis=0)
    sum_items = np.square(
        treatment_sums - 0.5 * n_subjects * (n_treatments + 1))
    q_stat = np.sum(sum_items, dtype=np.float64)
    tot = np.sum(ranked_data)

    for subj_idx in range(n_subjects):
        if len(np.unique(array[subj_idx, :])) != n_treatments:
            ties = True
            break
    else:
        ties = False

    if ties:
        if verbose:
            print(CONOVER_APPLIED)
        a1 = np.sum(np.square(ranked_data))
        c1 = 0.25 * n_subjects * n_treatments * (n_treatments + 1) ** 2
        # T1
        q_stat *= (n_treatments - 1) / (a1 - c1)

        # T2
        if t == 2:
            q_stat *= ((n_subjects - 1)
                       / (n_subjects * (n_treatments - 1) - q_stat))

    else:
        q_stat *= 12 / (n_subjects * n_treatments * (n_treatments + 1))

    return q_stat, tot


@nb.njit()
def test_faster(array: np.ndarray, tot):
    treatment_sums = pmu.np_sum(array[:, :-1], axis=0)
    sum_items = np.square(treatment_sums)
    q_stat = np.sum(sum_items, dtype=np.float64)
    q_stat = q_stat + np.square(tot - np.sum(treatment_sums))

    return q_stat
