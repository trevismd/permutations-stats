import numba as nb
import numpy as np
# noinspection PyPackageRequirements
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
# noinspection PyPackageRequirements
from scipy.stats import wilcoxon as scp_wilcoxon

import permutations_stats.permutations as pm
from permutations_stats.tests import wilcoxon

# noinspection DuplicatedCode
fd_array = np.array(
    [[103.3, 68.42, 89.53, 77.94, 100.0, 108.2, 184.9],
     [42.19, 44.31, 60.91, 73.90, 43.87, 61.74, 102.3],
     [71.27, 81.88, 100.71, 86.52, 100.3, 90.75, 100.6],
     [37.59, 60.05, 56.97, 60.79, 71.82, 83.04, 117.1],
     [58.31, 52.73, 96.04, 60.42, 104.33, 89.78, 85.77],
     [46.27, 82.18, 129.8, 181.0, 172.33, 164.8, 246.8],
     [19.00, 23.07, 38.70, 45.81, 59.03, 50.69, 55.18],
     [8.31, 8.43, 9.16, 14.28, 10.63, 15.84, 12.50],
     [20.15, 33.48, 60.08, 69.35, 114.34, 45.28, 101.6],
     [3.16, 4.12, 12.73, 18.95, 8.89, 41.52, 66.33],
     [4.12, 18.87, 8.54, 12.92, 25.30, 19.85, 16.76],
     [7.68, 11.18, 10.44, 10.95, 10.54, 13.96, 14.39],
     [5.29, 10.91, 11.22, 25.26, 42.25, 48.80, 69.38],
     [6.00, 5.57, 22.23, 2.45, 6.24, 1.00, 4.00]])


def test_repeated_permutations():
    array = np.arange(12).reshape((4, 3))
    res = pm.repeated_permutation_test(array)
    assert_equal(res[0], pm.tests.friedman.test(array))


def test_correct_alternative():
    array = np.arange(12).reshape((4, 3))

    with pytest.raises(ValueError):
        pm.repeated_permutation_test(array, alternative="this")


def test_repeated_permutations_simulation_override():
    array = np.arange(12).reshape((4, 3))
    res = pm.repeated_permutation_test(array)
    res2 = pm.repeated_permutation_test(array, method="simulation", n_iter=10000)
    assert_equal(res[0], res2[0])
    assert_almost_equal(res[1], res2[1])
    assert_equal(res[2], res2[2])


def test_repeated_permutations_simulation_number():
    array = np.arange(12).reshape((4, 3))
    res = pm.repeated_permutation_test(array, method="simulation", n_iter=100)
    assert_almost_equal(res[2], 101)


def test_friedman_permutation():
    res = pm.repeated_permutation_test(fd_array, method="simulation")
    assert round(res[0], 1) == 40.1


def test_repeated_permutations_w_seed(capsys):
    array = np.arange(12).reshape((4, 3))
    a = pm.repeated_permutation_test(array, method="simulation", n_iter=20,
                                     seed=100)[1]
    captured = capsys.readouterr()
    assert "Using seed" in captured.out
    assert_allclose(
        a,
        pm.repeated_permutation_test(array, method="simulation", n_iter=20,
                                     seed=100)[1])


def test_repeated_permutations_no_seed(capsys):
    array = np.repeat(np.arange(6).reshape((2, 3)), 2, axis=0)
    pm.repeated_permutation_test(array, method="simulation", n_iter=20)
    captured = capsys.readouterr()
    assert "Using seed" not in captured.out


def test_repeated_permutations_not_array():
    with pytest.raises(TypeError):
        pm.repeated_permutation_test("this")


@nb.njit()
def wilcoxon_scp(x, return_w=2.):
    # noinspection PyProtectedMember
    return wilcoxon._test(x, return_w)


@nb.njit()
def wilcoxon_scp_faster(x, return_w=2.):
    # noinspection PyProtectedMember
    return wilcoxon._test_faster(x, return_w)


def test_compare_exact_wilcoxon_no_ties():
    x = np.array([2, 4, 6, 7, 8, 9, 10, 12, 15])
    y = np.array([5.1, 5.2, 6.3, 6.4, 8.5, 10.6, 11.7, 13.9, 11])
    w_input = np.stack((x, y), axis=1)
    # using scipy's statistic doesn't yield same pvalue on permutations
    # So we compare separately
    wilcoxon_scp_dict = {"first": wilcoxon_scp, "then": wilcoxon_scp_faster}
    assert_allclose(
        pm.repeated_permutation_test(w_input, stat_func_dict=wilcoxon_scp_dict)[0],
        scp_wilcoxon(x, y, alternative="two-sided")[0]
    )
    assert_allclose(
        pm.repeated_permutation_test(w_input, test="wilcoxon")[1],
        scp_wilcoxon(x, y, alternative="two-sided")[1]
    )


def test_compare_exact_wilcoxon_no_ties_greater():
    x = np.array([2, 4, 6, 7, 8, 9, 10, 12, 15])
    y = np.array([5.1, 5.2, 6.3, 6.4, 8.5, 10.6, 11.7, 13.9, 11])
    w_input = np.stack((x, y), axis=1)
    # using scipy's statistic doesn't yield same pvalue on permutations
    # So we compare separately
    wilcoxon_scp_dict = {"first": wilcoxon_scp, "then": wilcoxon_scp_faster}
    assert_allclose(
        pm.repeated_permutation_test(
            w_input, stat_func_dict=wilcoxon_scp_dict, alternative="greater")[0],
        scp_wilcoxon(x, y, alternative="greater")[0]
    )
    assert_allclose(
        pm.repeated_permutation_test(w_input, test="wilcoxon", alternative="greater")[1],
        scp_wilcoxon(x, y, alternative="greater")[1]
    )


def test_compare_exact_wilcoxon_no_ties_less():
    x = np.array([2, 4, 6, 7, 8, 9, 10, 12, 15])
    y = np.array([5.1, 5.2, 6.3, 6.4, 8.5, 10.6, 11.7, 13.9, 11])
    w_input = np.stack((x, y), axis=1)
    # using scipy's statistic doesn't yield same pvalue on permutations
    # So we compare separately
    wilcoxon_scp_dict = {"first": wilcoxon_scp, "then": wilcoxon_scp_faster}
    assert_allclose(
        pm.repeated_permutation_test(w_input, stat_func_dict=wilcoxon_scp_dict, alternative="less")[0],
        scp_wilcoxon(x, y, alternative="less")[0]
    )
    assert_allclose(
        pm.repeated_permutation_test(w_input, test="wilcoxon", alternative="less")[1],
        scp_wilcoxon(x, y, alternative="less")[1]
    )
