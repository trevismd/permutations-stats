import itertools
import math
import random
import warnings
from collections import namedtuple

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
permutation_result = namedtuple("PermutationsResults",
                                ("statistic", "pvalue", "permutations",
                                 "test", "alternative", "method"))


def friedman(array, rank=True):
    n, k = array.shape
    if rank:
        new_array = pmu.simpler_ranks(array)
    else:
        new_array = array
    critical_value = np.sum(np.square(np.sum(new_array, 0)))

    tot = np.sum(new_array)
    s_indices = [indices[:-1] for indices in itertools.permutations(range(k))]
    fs_max_tot = n * k
    fs = np.zeros([fs_max_tot for _ in range(k-1)])

    row = new_array[0]
    perm_values = row.take(s_indices)

    for coords in perm_values:
        fs[tuple(coords)] += 1

    axes = tuple(range(k-1))

    for i in range(1, n):
        row = new_array[i]
        perm_values = row.take(s_indices)
        fs_new = np.zeros(fs.shape)
        for perm_idx, coords in enumerate(perm_values):
            fs_new += np.roll(fs, coords, axis=axes)
        fs = fs_new

    non_zero = np.nonzero(fs)
    counts = np.empty((len(non_zero[0]), 2))

    for val_idx, values in enumerate(zip(*non_zero)):
        counts[val_idx] = fs[values], np.sum(np.square([*values,
                                                        tot - np.sum(values)]))
    tot_perms = np.sum(fs)

    pvalue = (np.sum(np.where(counts[:, 1] >= critical_value, counts[:, 0], 0))
              / tot_perms)

    return pvalue


def permutation_test(x: np.array, y: np.array, test="brunner_munzel",
                     stat_func_dict=None, alternative="two-sided",
                     method="exact", n_iter=1e4, force_simulations=False,
                     seed=None) -> permutation_result:

    alternative, stat_func_first, stat_func_then, method = _check_and_get(
        test, alternative, stat_func_dict, method)

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

    w_and_params = stat_func_first(x, y)
    w, args = w_and_params

    comp_w = stat_func_then(x, y, args)

    n_all_comb = (math.factorial(n_tot)
                  // math.factorial(n_x)
                  // math.factorial(n_y))

    method = _check_method(n_iter, n_all_comb, method, force_simulations)

    if method == pmu.Method.exact:
        n_comb = n_all_comb
        perm_ids = [perm_x_idx
                    for perm_x_idx
                    in itertools.combinations(range(n_tot), n_x)]

        perm_ids = np.array(perm_ids, dtype=np.int32)

        res_greater, res_smaller, res_equal = _get_counts(
            all_values, perm_ids, comp_w, stat_func_then, args)

    else:
        n_comb = n_iter

        if seed is not None:
            random.seed(seed)

        perm_ids = [sorted(random.sample(range(n_tot), n_x))
                    for _ in range(n_iter)]
        perm_ids = np.array(perm_ids, dtype=np.int32)

        res_greater, res_smaller, res_equal = _get_counts(
            all_values, perm_ids, comp_w, stat_func_then, args)

        # Be sure to have at least one possibility
        res_equal += 1
        n_comb += 1

    pval = _compute_pval(
        res_greater, res_smaller, res_equal, n_comb, alternative)

    return permutation_result(
        w, pval, n_comb, test, alternative.name, method.name)


def repeated_permutation_test(x: np.array, test="friedman",
                              stat_func_dict=None, alternative="two-sided",
                              method="exact", n_iter=1e4,
                              force_simulations=False, seed=None):
    try:
        x = x.astype(np.float64)

    except:  # Must be simplified for numba support  # noqa: E722
        raise TypeError("Please provide numeric valued numpy arrays (or "
                        "-like, np.array 1st argument) for x.")

    if len(x.shape) != 2:
        raise ValueError("Please provide a 2D array(-like) for x.")

    n_subjects, n_treatments = x.shape
    alternative, stat_func_first, stat_func_then, method = _check_and_get(
        test, alternative, stat_func_dict, method)

    n_iter = int(n_iter)

    w, args = stat_func_first(x)
    try:
        transform = (
            TESTS.get(test, None).get(alternative, None).get("transform", None)
        )

    except AttributeError:
        transform = None

    if transform is not None:
        x = transform(x)

    comp_w = stat_func_then(x, args)  # to be used for comparisons

    n_all_comb = math.factorial(n_treatments) ** n_subjects
    method = _check_method(n_iter, n_all_comb, method, force_simulations)
    treatment_perms_ids = list(itertools.permutations(range(n_treatments)))

    treatment_perms_ids = np.array(treatment_perms_ids, dtype=np.int32)

    if method == pmu.Method.exact:
        n_comb = n_all_comb

        res_greater, res_smaller, res_equal = _all_dependent(
            x, n_comb, treatment_perms_ids, comp_w, stat_func_then, args)

    else:
        n_comb = n_iter

        res_greater, res_smaller, res_equal = _simulate_dependent(
            x, treatment_perms_ids, n_iter, comp_w, stat_func_then, args, seed)

        # Be sure to have at least one possibility
        res_equal += 1
        n_comb += 1

    pval = _compute_pval(
        res_greater, res_smaller, res_equal, n_comb, alternative)

    return permutation_result(
        w, pval, n_comb, test, alternative.name, method.name)


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


# noinspection DuplicatedCode
@nb.njit(parallel=True)
def _all_dependent(array, n_comb, treatment_perms_ids, w, func, args):
    res_greater, res_smaller, res_equal = (0, 0, 0)
    n_subjects = array.shape[0]
    n_trt_perms = len(treatment_perms_ids)

    for value in nb.prange(n_comb):
        new_array = np.empty_like(array)

        subj_perms_ids = num_to_array(value, n_trt_perms, n_subjects)
        for row_idx, val in enumerate(subj_perms_ids):
            new_array[row_idx, :] = array[row_idx].take(
                treatment_perms_ids[val])

        res_i = func(new_array, args)

        if pmu.is_close(res_i, w):
            res_equal += 1

        elif res_i > w:
            res_greater += 1

        else:
            res_smaller += 1

    return res_greater, res_smaller, res_equal


# No parallel if seed is set so splitting of this function could be useful
# noinspection DuplicatedCode
@nb.njit()
def _simulate_dependent(array, treatment_perms_ids, n_iter, w,
                        func, args, seed):
    res_greater, res_smaller, res_equal = (0, 0, 0)
    new_array = np.empty_like(array)
    n_subjects = array.shape[0]
    n_trt_perms = len(treatment_perms_ids)

    if seed is not None:
        np.random.seed(seed)
        print("Using seed", seed)

    for _ in range(n_iter):
        subj_perms_ids = np.random.choice(
            n_trt_perms, n_subjects, replace=True)

        for row_idx, val in enumerate(subj_perms_ids):
            new_array[row_idx, :] = array[row_idx].take(
                treatment_perms_ids[val])

        res_i = func(new_array, args)

        if pmu.is_close(res_i, w):
            res_equal += 1

        elif res_i > w:
            res_greater += 1

        else:
            res_smaller += 1

    return res_greater, res_smaller, res_equal


def _check_and_get(test, alternative, stat_func_dict, method):
    global ALTERNATIVES, TESTS

    alternative = ALTERNATIVES.get(alternative, None)

    if alternative is None:
        raise ValueError(f"Incorrect `alternative` specified, "
                         f"must be one of {list(ALTERNATIVES.keys())}")

    if stat_func_dict is None:
        if not isinstance(test, str):
            raise TypeError("`test` parameter must be a string")

        stat_func_dicts = TESTS.get(test, None)

        if stat_func_dicts is not None:
            stat_func_dict = stat_func_dicts.get(alternative, None)

        if stat_func_dicts is None or stat_func_dict is None:
            raise ValueError(f"Incorrect `test` name specified, "
                             f"must be one of {list(TESTS.keys())} "
                             f"or a function must be passed as `stat_func`")
        else:
            stat_func_first = stat_func_dict["first"]
            stat_func_then = stat_func_dict["then"]

    else:
        try:
            stat_func_first = stat_func_dict["first"]
            stat_func_then = stat_func_dict["then"]

        except IndexError:
            raise IndexError("stat_func_dict expects a mapping of "
                             "'first' and 'then' to callables.")

        if not callable(stat_func_first) or not callable(stat_func_then):
            raise TypeError("stat_func must be a callable object (function)")

    method = METHODS.get(method, None)

    if method is None:
        raise ValueError(f"Incorrect computation method specified, "
                         f"must be in {list(METHODS.keys())}")

    return alternative, stat_func_first, stat_func_then, method


@nb.njit()
def _compute_pval(res_greater, res_smaller, res_equal, n_comb, alternative):
    if alternative == pmu.Alternative.LESS:
        pval = (res_smaller + res_equal) / n_comb

    else:
        pval = (res_greater + res_equal) / n_comb

    return pval


@nb.njit(parallel=True)
def _get_counts(values, perm_ids, w, func, args):
    res_equal, res_greater, res_smaller = (0, 0, 0)

    for i in nb.prange(len(perm_ids)):
        perm_x_idx = perm_ids[i]
        new_x = np.take(values, perm_x_idx)
        new_y = np.delete(values, perm_x_idx)
        res_i = func(new_x, new_y, args)

        if pmu.is_close(res_i, w):
            res_equal += 1

        elif res_i > w:
            res_greater += 1

        else:
            res_smaller += 1

    return res_greater, res_smaller, res_equal


def _check_method(n_iter, n_all_comb, method, force_simulations):
    if n_iter + 1 >= n_all_comb and method == pmu.Method.simulation \
            and not force_simulations:

        warnings.warn(
            f"Simulation overridden by exact test because total number of "
            f"combinations ({n_all_comb}) is smaller than asked amount of "
            f"simulation iterations ({n_iter}).\n"
            f"Pass `force_simulations=True` to avoid this behavior")

        method = pmu.Method.exact
    return method
