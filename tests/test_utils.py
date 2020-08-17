import numpy as np

from permutations_stats.utils import rank_array
from permutations_stats.utils import is_close


def test_simple_rank():
    r1 = np.array([1, 0, 8, 6, 3, 2, 33, 9])
    assert np.allclose(rank_array(r1),
                          np.array([2, 1, 6, 5, 4, 3, 8, 7]))


def test_is_close():
    a = 2.000000000000001
    b = 3. - 1.
    assert a != b
    assert is_close(a, b)
