import itertools
import math
import random

import numba as nb
import numpy as np

import permutations_stats.tests.tests as tests
import permutations_stats.utils as pmu

ALTERNATIVES = {"two-sided": pmu.Alternative.TWO_SIDED,
                "t": pmu.Alternative.TWO_SIDED,
                "d": pmu.Alternative.TWO_SIDED,
                "greater": pmu.Alternative.GREATER,
                "g": pmu.Alternative.GREATER,
                "less": pmu.Alternative.LESS,
                "l": pmu.Alternative.LESS}

METHODS = {"exact": pmu.Method.exact,
           "simulation": pmu.Method.simulation,
           "approximate": pmu.Method.simulation}


TESTS = tests.TESTS


def permutation_test(x: np.array, y: np.array, test="brunner_munzel",
                     stat_func=None, alternative="two-sided", method="exact",
                     n_iter=1e4, force_simulations=False, seed=None):

    alternative, stat_func, method = _check_and_get(test, alternative, stat_func,
                                                    method)

    n_iter = int(n_iter)

    try:
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

    except (TypeError, ValueError) as e:
        raise e("Please provide numeric valued numpy arrays (or -like, "
                "np.array 1st argument) for x and y.")

    all_values = np.concatenate((x, y))

    n_x = len(x)
    n_y = len(y)
    n_tot = n_x + n_y

    w = stat_func(x, y)

    comp_w = abs(w) if alternative == pmu.Alternative.TWO_SIDED else w  # to be used for comparisons

    n_all_comb = math.factorial(n_tot) // math.factorial(n_x) // math.factorial(n_y)

    method = _check_method(n_iter, n_all_comb, method, force_simulations)

    if method == pmu.Method.exact:
        n_comb = n_all_comb
        perm_ids = [perm_x_idx
                    for perm_x_idx in itertools.combinations(range(n_tot), n_x)]
        perm_ids = np.array(perm_ids, dtype=np.int32)

        res_greater, res_smaller, res_equal = _get_counts(
            all_values, perm_ids, comp_w, alternative, stat_func)

    else:
        n_comb = n_iter

        if seed is not None:
            random.seed(seed)

        perm_ids = [sorted(random.sample(range(n_tot), n_x))
                    for _ in range(n_iter)]
        perm_ids = np.array(perm_ids, dtype=np.int32)

        res_greater, res_smaller, res_equal = _get_counts(
            all_values, perm_ids, comp_w, alternative, stat_func)

        # Be sure to have at least one possibility
        res_equal += 1
        n_comb += 1

    pval = _compute_pval(res_greater, res_smaller, res_equal, n_comb, alternative)

    return w, pval, n_comb


def repeated_permutation_test(x: np.array, test="friedman",
                              stat_func=None, alternative="two-sided",
                              method="exact", n_iter=1e4,
                              force_simulations=False, seed=None):
    try:
        x = x.astype(np.float64)

    except:  # Must be simpliflied for numba support
        raise TypeError("Please provide numeric valued numpy arrays (or -like, "
                        "np.array 1st argument) for x.")

    n_subjects, n_treatments = x.shape
    alternative, stat_func, method = _check_and_get(test, alternative, stat_func,
                                                    method)
    n_iter = int(n_iter)

    w = stat_func(x)
    comp_w = abs(w) if alternative == pmu.Alternative.TWO_SIDED else w  # to be used for comparisons

    n_all_comb = math.factorial(n_treatments) ** n_subjects
    method = _check_method(n_iter, n_all_comb, method, force_simulations)
    treatment_perms_ids = [perm_x_idx
                           for perm_x_idx
                           in itertools.permutations(range(n_treatments))]

    treatment_perms_ids = np.array(treatment_perms_ids, dtype=np.int32)

    if method == pmu.Method.exact:
        n_comb = n_all_comb

        res_greater, res_smaller, res_equal = _all_dependent(
            x, n_comb, treatment_perms_ids, comp_w, stat_func, alternative)

    else:
        n_comb = n_iter

        res_greater, res_smaller, res_equal = _simulate_dependent(
            x, treatment_perms_ids, n_iter, comp_w, stat_func, alternative, seed)

        # Be sure to have at least one possibility
        res_equal += 1
        n_comb += 1

    pval = _compute_pval(res_greater, res_smaller, res_equal, n_comb, alternative)

    return w, pval, n_comb


@nb.njit()
def num_to_array(value, n_values, n_elem):
    pem = np.zeros((n_elem,), dtype=np.int64)
    for elem_idx in range(n_elem):
        excess = value % (n_values ** (elem_idx + 1))
        value -= excess
        pem[-elem_idx-1] = excess / (n_values ** elem_idx)
        if value == 0:
            return pem
    return pem


@nb.njit(parallel=True)
def _all_dependent(array, n_comb, treatment_perms_ids, w, func, alternative):
    res_greater, res_smaller, res_equal = (0, 0, 0)
    new_array = np.empty_like(array)
    n_subjects = array.shape[0]
    n_trt_perms = len(treatment_perms_ids)

    # for subj_perms_ids in gen_values_n(n_trt_perms, n_subjects):
    for value in range(n_comb):
        subj_perms_ids = num_to_array(value, n_trt_perms, n_subjects)
        for row_idx, val in enumerate(subj_perms_ids):
            new_array[row_idx, :] = array[row_idx].take(treatment_perms_ids[val])

        res_i = func(new_array)

        if alternative == pmu.Alternative.TWO_SIDED:
            res_i = abs(res_i)

        if pmu.is_close(res_i, w):
            res_equal += 1

        elif res_i > w:
            res_greater += 1

        else:
            res_smaller +=1

    return res_greater, res_smaller, res_equal


@nb.njit()
def _simulate_dependent(array, treatment_perms_ids, n_iter, w,
                        func, alternative, seed):
    res_greater, res_smaller, res_equal = (0, 0, 0)
    new_array = np.empty_like(array)
    n_subjects = array.shape[0]
    n_trt_perms = len(treatment_perms_ids)

    if seed is not None:
        np.random.seed(seed)

    for _ in range(n_iter):
        subj_perms_ids = np.random.choice(n_trt_perms, n_subjects, replace=True)

        for row_idx, val in enumerate(subj_perms_ids):
            new_array[row_idx, :] = array[row_idx].take(treatment_perms_ids[val])

        res_i = func(new_array)

        if alternative == pmu.Alternative.TWO_SIDED:
            res_i = abs(res_i)

        if pmu.is_close(res_i, w):
            res_equal += 1

        elif res_i > w:
            res_greater += 1

        else:
            res_smaller +=1

    return res_greater, res_smaller, res_equal


def _check_and_get(test, alternative, stat_func, method):
    global ALTERNATIVES, TESTS

    alternative = ALTERNATIVES.get(alternative, None)

    if alternative is None:
        raise ValueError(f"Incorrect `alternative` specified, "
                         f"must be one of {list(ALTERNATIVES.keys())}")

    if stat_func is None:
        if not isinstance(test, str):
            raise TypeError(f"`test` parameter must be a string")

        stat_func = TESTS.get(test, None)

        if stat_func is None:
            raise ValueError(f"Incorrect `test` name specified, "
                             f"must be one of {list(TESTS.keys())} "
                             f"or a function must be passed as `stat_func`")

    elif not callable(stat_func):
        raise TypeError("stat_func must be a callable object (function)")

    method = METHODS.get(method, None)

    if method is None:
        raise ValueError(f"Incorrect computation method specified, "
                         f"must be in {list(METHODS.keys())}")

    return alternative, stat_func, method


@nb.njit()
def _compute_pval(res_greater, res_smaller, res_equal, n_comb, alternative):
    if alternative == pmu.Alternative.LESS:
        pval = (res_smaller + res_equal) / n_comb

    else:
        pval = (res_greater + res_equal) / n_comb

    return pval


@nb.njit(parallel=True)
def _get_counts(values, perm_ids, w, alternative, func):
    res_equal, res_greater, res_smaller = (0, 0, 0)

    for i, perm_x_idx in enumerate(perm_ids):
        new_x = np.take(values, perm_x_idx)
        new_y = np.delete(values, perm_x_idx)
        res_i = func(new_x, new_y)

        if alternative == pmu.Alternative.TWO_SIDED:
            res_i = abs(res_i)

        if pmu.is_close(res_i, w):
            res_equal += 1

        elif res_i > w:
            res_greater += 1

        else:
            res_smaller +=1

    return res_greater, res_smaller, res_equal


def _check_method(n_iter, n_all_comb, method, force_simulations):
    if n_iter + 1 >= n_all_comb and method == pmu.Method.simulation \
            and not force_simulations:

        print(f"Simulation overridden by exact test because total number of "
              f"combinations ({n_all_comb}) is smaller than asked amount of "
              f"simulation iterations ({n_iter}).\n"
              f"Pass `force_simulations=True` to avoid this behavior")

        method = pmu.Method.exact
    return method
