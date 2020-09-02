import math

import numba as nb
import numpy as np

import permutations_stats.utils as pmu


@nb.njit()
def test(array: np.ndarray):
    """
    Subjects are array first dimension.
    Treatments are should be the array second condition.
    In case of ties Conover's statistic is calculated and T2 is returned instead
    of T1 if True
    :params array: Data in shape (n_subjects, n_treamtments)
    :return Friedman statistic
    """
    if len(array.shape) != 2:
        raise ValueError("Input array should be 2d")

    n_subjects = array.shape[0]
    n_treatments = array.shape[1]

    ranked_data = pmu.rank_2d_by_row(array)
    treatment_means = pmu.np_mean(ranked_data, axis=0)
    p_item = (n_treatments + 1) * -0.5
    sum_items = np.square(treatment_means - 0.5 * (n_treatments +1))
    Q = np.sum(sum_items)

    for subj_idx in range(n_subjects):
        if len(np.unique(array[subj_idx, :])) != n_treatments:
            ties = True
    else:
        ties = False

    if ties:
        print("Conover correction for ties applied")
        a1 = np.sum(np.square(ranked_data))
        c1 = 0.25 * n_subjects * n_treatments * (n_treatments + 1) ** 2
        Q *= (n_treatments - 1) / (a1 - c1)
        if True:  # Conover
            Q *= (n_subjects - 1) / ( n_subjects * (n_treatments - 1) - Q)
    else:
        Q *= 12 * n_subjects / (n_treatments * (n_treatments + 1))

    return Q
