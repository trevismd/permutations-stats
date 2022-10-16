import numba as nb
import numpy as np


def bootstrap_mean(values, width=95, n_iter=1000, seed=0, nb_threshold=2e5,
                   verbose=False):
    """
    Resamples with replacement to provide a confidence interval for the mean.
    np.mean and np.percentiles are used with default parameters.
    :param values: 1D array
    :param width: Width of confidence interval (symmetric).
        Default is 95 for 95% CI
    :param n_iter: Number of iterations to perform. Default is 1000.
    :param seed: Seed for the random sampling, to provide reproducible results.
    :param nb_threshold: Number of operations before using numba.
    :param verbose: prints if numba was used.
    :return: mean of sample, lower bound, upper bound.
    """
    if seed or n_iter * len(values) < nb_threshold:
        if verbose:
            print("Using numpy only")

        return _bootstrap_numpy(np.mean, values, width, n_iter, seed)

    if verbose:
        print("Using numba")

    return _bootstrap_mean_pll(values, width, n_iter)


def bootstrap_median(values, width=95, n_iter=1000, seed=0, nb_threshold=2e5,
                     verbose=False):
    """
    Resamples with replacement to provide a confidence interval for the median.
    np.median and np.percentiles are used with default parameters.

    :param values: 1D array
    :param width: Width of confidence interval (symmetric).
        Default is 95 for 95% CI
    :param n_iter: Number of iterations to perform. Default is 1000.
    :param seed: Seed for the random sampling, to provide reproducible results.
    :param nb_threshold: Number of operations before using numba.
    :param verbose: prints if numba was used.
    :return: median of sample, lower bound, upper bound.
    """
    if seed or n_iter * len(values) < nb_threshold:
        if verbose:
            print("Using numpy only")

        return _bootstrap_numpy(np.median, values, width, n_iter, seed)

    if verbose:
        print("Using numba")

    return _bootstrap_median_pll(values, width, n_iter)


def bootstrap_std(values, width=95, n_iter=1000, seed=0, nb_threshold=2e5,
                  verbose=False):
    """
    Resamples with replacement to provide a confidence interval for the
    standard deviation.
    np.std and np.percentiles are used with default parameters.

    :param values: 1D array
    :param width: Width of confidence interval (symmetric).
        Default is 95 for 95% CI
    :param n_iter: Number of iterations to perform. Default is 1000.
    :param seed: Seed for the random sampling, to provide reproducible results.
    :param nb_threshold: Number of operations before using numba.
    :param verbose: prints if numba was used.
    :return: standard deviation of sample, lower bound, upper bound.
    """
    if seed or n_iter * len(values) < nb_threshold:
        if verbose:
            print("Using numpy only")

        return _bootstrap_numpy(np.std, values, width, n_iter, seed)

    if verbose:
        print("Using numba")

    return _bootstrap_std_pll(values, width, n_iter)


def _bootstrap_numpy(func, values, width=95, n_iter=1000, seed=0):
    """
    Resamples with replacement to provide a confidence interval for the mean.
    np.mean and np.percentiles are used with default parameters.
    :param values: 1D array
    :param width: Width of confidence interval (symmetric).
        Default is 95 for 95% CI
    :param n_iter: Number of iterations to perform. Default is 1000.
    :param seed: Seed for the random sampling, to provide reproducible results.
    :return: mean of sample, lower bound, upper bound.
    """
    n_items = values.shape[0]
    results = np.empty(n_iter, dtype=np.float64)

    if seed != 0:
        np.random.seed(seed)

    for iter_idx in range(n_iter):
        results[iter_idx] = func(
            values.take(np.random.randint(0, n_items, n_items)))

    res = func(values)

    return _shape_results(res, results, width)


@nb.njit(parallel=True)
def _bootstrap_mean_pll(values, width=95, n_iter=1000):
    """
    Resamples with replacement to provide a confidence interval for the mean.
    np.mean and np.percentiles are used with default parameters.
    :param values: 1D array
    :param width: Width of confidence interval (symmetric).
        Default is 95 for 95% CI
    :param n_iter: Number of iterations to perform. Default is 1000.
    :return: mean of sample, lower bound, upper bound.
    """
    n_items = values.shape[0]
    results = np.empty(n_iter, dtype=np.float64)

    for iter_idx in range(n_iter):
        results[iter_idx] = np.mean(
            values.take(np.random.randint(0, n_items, n_items)))

    res = np.mean(values)

    return _shape_results(res, results, width)


@nb.njit(parallel=True)
def _bootstrap_median_pll(values, width=95, n_iter=1000):
    """
    Resamples with replacement to provide a confidence interval for the median.
    np.median and np.percentiles are used with default parameters.

    :param values: 1D array
    :param width: Width of confidence interval (symmetric).
        Default is 95 for 95% CI
    :param n_iter: Number of iterations to perform. Default is 1000.
    :return: median of sample, lower bound, upper bound.
    """

    n_items = values.shape[0]
    results = np.empty(n_iter, dtype=np.float64)

    for iter_idx in range(n_iter):
        results[iter_idx] = np.median(
            values.take(np.random.randint(0, n_items, n_items)))

    res = np.median(values)

    return _shape_results(res, results, width)


@nb.njit(parallel=True)
def _bootstrap_std_pll(values, width=95, n_iter=1000):
    """
    Resamples with replacement to provide a confidence interval for the
    standard deviation.
    np.std and np.percentiles are used with default parameters.

    :param values: 1D array
    :param width: Width of confidence interval (symmetric).
        Default is 95 for 95% CI
    :param n_iter: Number of iterations to perform. Default is 1000.
    :return: standard deviation of sample, lower bound, upper bound.
    """
    n_items = values.shape[0]
    results = np.empty(n_iter, dtype=np.float64)

    for iter_idx in range(n_iter):
        results[iter_idx] = np.std(
            values.take(np.random.randint(0, n_items, n_items)))

    res = np.std(values)

    return _shape_results(res, results, width)


@nb.njit()
def _shape_results(res, results, width):
    ordered_results = np.sort(results)

    lower = np.percentile(ordered_results, (100 - width) / 2)
    upper = np.percentile(ordered_results, (100 + width) / 2)

    return res, lower, upper
