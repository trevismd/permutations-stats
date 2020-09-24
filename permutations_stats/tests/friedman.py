import numba as nb
import numpy as np

import permutations_stats.utils as pmu

CONOVER_APPLIED = "Conover correction for ties applied"


# @nb.njit(['float64(float64[:, :])',
#           'float64(float32[:, :])',
#           'float64(int32[:, :])',
#           'float64(int64[:, :])'])
@nb.njit()
def test(array: np.ndarray, verbose=False):
    """
    Subjects are array first dimension.
    Treatments are the array second condition.
    In case of ties Conover's statistic is calculated and T2 is returned instead
    of T1.
    :params array: Data in shape (n_subjects, n_treatments)
    :return Friedman statistic
    """
    # noinspection PyBroadException
    try:
        if len(array.shape) != 2:
            raise NotImplementedError("Input should be a 2D array.")

    except:
        raise TypeError("Please provide a numeric-valued 2D numpy arrays for x.")

    n_subjects = array.shape[0]
    n_treatments = array.shape[1]

    ranked_data = pmu.rank_2d_by_row(array)
    treatment_means = pmu.np_mean(ranked_data, axis=0)
    sum_items = np.square(treatment_means - 0.5 * (n_treatments + 1))
    q_stat = np.sum(sum_items, dtype=np.float64)

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
        q_stat *= (n_treatments - 1) / (a1 - c1)

        q_stat *= ((n_subjects - 1)
                   / (n_subjects * (n_treatments - 1) - q_stat))

    else:
        q_stat *= 12 * n_subjects / (n_treatments * (n_treatments + 1))

    return q_stat


@nb.njit()
def test_faster(array: np.ndarray, verbose=False):
    """
    Subjects are array first dimension.
    Treatments are the array second condition.
    In case of ties Conover's statistic is calculated and T2 is returned instead
    of T1.
    :params array: Data in shape (n_subjects, n_treatments)
    :return Friedman statistic
    """
    # noinspection PyBroadException

    ranked_data = pmu.rank_2d_by_row(array)
    treatment_means = pmu.np_sum(ranked_data, axis=0)
    sum_items = np.square(treatment_means)
    q_stat = np.sum(sum_items, dtype=np.float64)

    return q_stat
