import itertools
import math
import random

import numba as nb
import numpy as np

from permutations_stats.tests import tests
from permutations_stats.utils import is_close

ALTERNATIVES = {"two-sided": 1,
                "t": 1,
                "d": 1,
                "greater": 2,
                "g": 2,
                "less": 3,
                "l": 3}

METHODS = ["exact", "simulation", "approximate"]
TESTS = tests.TESTS


@nb.njit()
def get_counts(values, perm_ids, w, alt, func):
    res = np.empty(len(perm_ids), dtype=np.float64)

    for i, perm_x_idx in enumerate(perm_ids):
        new_x = np.take(values, perm_x_idx)
        new_y = np.delete(values, perm_x_idx)
        res[i] = func(new_x, new_y)

    if alt == 1:
        res = np.abs(res)

    close = is_close(res, w)

    res_equal = np.sum(close)
    res_greater = np.sum(res[~close] > w)
    res_smaller = np.sum(res[~close] < w)

    return res_greater, res_smaller, res_equal


def permutation_test(x: np.array, y: np.array, test="brunner_munzel",
                     stat_func=None, alternative="two-sided", method="exact",
                     n_iter=1e4, force_simulations=False, seed=None):

    global ALTERNATIVES, TESTS
    n_iter = int(n_iter)
    alternative = ALTERNATIVES.get(alternative, None)
    if alternative is None:
        raise ValueError(f"Incorrect `alternative` specified, "
                         f"must be one of {list(ALTERNATIVES.keys())}")

    if stat_func is None:
        stat_func = TESTS.get(test, None)
        if stat_func is None:
            raise ValueError(f"Incorrect `test` name specified, "
                             f"must be one of {list(TESTS.keys())} "
                             f"or a function must be passed as `stat_func`")

    if method not in METHODS:
        raise ValueError(f"Incorrect computation method specified, "
                         f"must be in {METHODS}")

    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
            y = np.array(y)

        except TypeError:
            raise TypeError("Please provide numpy arrays "
                            "(or np.array 1st arguments) for x and y.")

    all_values = np.concatenate((x, y))

    n_x = len(x)
    n_y = len(y)
    n_tot = n_x + n_y

    w = stat_func(x, y)

    comp_w = abs(w) if alternative == 1 else w  # to be used for comparisons

    n_all_comb = math.factorial(n_tot) // math.factorial(n_x) // math.factorial(n_y)

    if n_iter + 1 >= n_all_comb \
            and (method == "simulation" or method == "approximate") \
            and not force_simulations:

        print(f"Simulation overridden by exact test because total number of "
              f"combinations ({n_all_comb}) is smaller than asked amount of "
              f"simulation iterations ({n_iter}).\n"
              f"Pass `force_simulations=True` to avoid this behavior")

        method = "exact"

    if method == "exact":
        n_comb = n_all_comb
        perm_ids = [perm_x_idx
                    for perm_x_idx in itertools.combinations(range(n_tot), n_x)]
        perm_ids = np.array(perm_ids, dtype=np.int32)

        res_greater, res_smaller, res_equal = get_counts(
            all_values, perm_ids, comp_w, alternative, stat_func)

    else:
        n_comb = n_iter

        if seed is None:
            seed = n_iter

        random.seed(seed)
        perm_ids = [sorted(random.sample(range(n_tot), n_x))
                    for _ in range(n_iter)]
        perm_ids = np.array(perm_ids, dtype=np.int32)

        res_greater, res_smaller, res_equal = get_counts(
            all_values, perm_ids, comp_w, alternative, stat_func)

        # Be sure to have at least one possibility
        res_equal += 1
        n_comb += 1

    if alternative == 2:
        pval = (res_smaller + res_equal) / n_comb

    else:
        pval = (res_greater + res_equal) / n_comb

    return w, pval, n_comb
