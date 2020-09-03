import numpy as np
import random
import numba as nb


@nb.njit()
def bootstrap_mean(values, alpha, n_iter, seed: int = 0):
    n_items = values.shape[0]
    results = np.empty(n_iter, dtype=np.float64)

    if seed > 0:
        random.seed(seed)

    for iter_idx in range(n_iter):
        results[iter_idx] = np.mean(values.take(list(
            [random.randrange(n_items) for _ in range(n_items)])))

    res = np.mean(values)

    return shape_results(res, results, alpha)


@nb.njit()
def bootstrap_median(values, alpha, n_iter, seed: int = 0):
    n_items = values.shape[0]
    results = np.empty(n_iter, dtype=np.float64)

    if seed > 0:
        random.seed(seed)

    for iter_idx in range(n_iter):
        results[iter_idx] = np.median(values.take(list(
            [random.randrange(n_items) for _ in range(n_items)])))

    res = np.median(values)

    return shape_results(res, results, alpha)


@nb.njit()
def shape_results(res, results, alpha):
    ordered_results = np.sort(results)

    lower = np.percentile(ordered_results, 100 * alpha / 2)
    upper = np.percentile(ordered_results, 100 - (100 * alpha / 2))

    return res, lower, upper
