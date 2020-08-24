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

    ordered_results = np.sort(results)

    lower = np.percentile(ordered_results, 100 * alpha / 2)
    upper = np.percentile(ordered_results, 100 - (100 * alpha / 2))
    mean_ = np.mean(values)

    return mean_, lower, upper
