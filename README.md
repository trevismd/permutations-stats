# permutations-stats
Permutation-based statistical package providing exact tests (for small samples).  
Accelerated with numba. 
# Status
Brunner Munzel is the only test implemented yet.   
Exact testing (all permutations) and an approximate method (simulation) are available.

In the current implementation, a few seconds are
required to compile on the fly.
Then, acceleration is critical as shown in this output from tests comparing the 
statistic calculation with scipy:

```
tests/stat_tests.py
200 tests with 5 and 4 data points - Permutations-stats: 2.375 s, Scipy: 0.076s, diff: -2.299
.
200 tests with 3 and 3 data points - Permutations-stats: 0.002 s, Scipy: 0.077s, diff: 0.075
.
200 tests with 10 and 12 data points - Permutations-stats: 0.004 s, Scipy: 0.079s, diff: 0.075
.
10000 tests with 18 and 10 data points - Permutations-stats: 0.220 s, Scipy: 3.806s, diff: 3.586
.
20000 tests with 18 and 15 data points - Permutations-stats: 0.477 s, Scipy: 7.645s, diff: 7.168
.
30000 tests with 18 and 19 data points - Permutations-stats: 0.769 s, Scipy: 11.468s, diff: 10.699
```

## Usage
```python
import numpy as np
from permutations_stats.permutations import permutation_test

# Sample data
x = np.arange(9)
y = (np.arange(8) -0.2) * 1.1
```
Default parameters are exact Brunner Munzel test (two-sided)

```python
stat, pval = permutation_test(x, y, test="brunner_munzel")
stat, pval
# (-0.2776044311308564, 0.7475935828877005)
```

Simulations are run like this (n_iter has default 100 000 iterations and can be omitted)
```python
_, pval = permutation_test(x, y, method="approximate", n_iter = 1e4)
# pval
# 0.7409259074092591
```

Other alternatives are possible (`"greater"` and `"less"`, default is `"two-sided"`)
```python
_, pval = permutation_test(x, y, alternative="greater")
# pval
# 0.37379679144385025
```
