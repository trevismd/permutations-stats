![coverage](.github/coverage.svg)
# permutations-stats
Python only permutation-based statistical tests, accelerated with numba.
## Status
### Statistical tests
Brunner Munzel [1] and Friedman [2] tests (repeated measures) are implemented.

Exact testing (all permutations) and approximate method (simulation) are available.

Two functions for bootstrapping-based confidence intervals calculations for 
mean and median are also implemented (not documented yet).

## Why this package ?
This work aims to provide fast permutation-based statistical tests in Python.
Some tests are not available publicly in an exact mode (computing all 
possible permutations) or with simulations. So if certain assumptions cannot be 
made about the data (such as normality or a sufficiently large sample), these 
implementations shouldn't be used.
For example, the Brunner Munzel [1] test is implemented in scipy but not with 
an exact calculation. The statistic can be used with the public API but it can 
take some time ran thousands of times (the p-value is also calculated for each 
iteration).  

This packages reimplements the looping of permutations and statistical tests 
with numba. With numba a few seconds are required to compile on the fly.  
Then, acceleration is critical as shown in this output from tests comparing the
Brunner Munzel statistic calculation with scipy:

```
tests/stat_tests.py
200 tests with 5 and 4 data points - Permutations-stats: 2.375s, Scipy: 0.076s, diff: -2.299s
.
200 tests with 3 and 3 data points - Permutations-stats: 0.002s, Scipy: 0.077s, diff: 0.075s
.
200 tests with 10 and 12 data points - Permutations-stats: 0.004s, Scipy: 0.079s, diff: 0.075s
.
10000 tests with 18 and 10 data points - Permutations-stats: 0.220s, Scipy: 3.806s, diff: 3.586s
.
20000 tests with 18 and 15 data points - Permutations-stats: 0.477s, Scipy: 7.645s, diff: 7.168s
.
30000 tests with 18 and 19 data points - Permutations-stats: 0.769s, Scipy: 11.468s, diff: 10.699s
```

Scipy is working on numba implementation too, but these functions are not 
available at the moment.

## Dependencies
* numpy
* numba

And for development testing only
* scipy, pytest

## Usage
Basically,
```python
import numpy as np
from permutations_stats.permutations import permutation_test

# Sample data
x = np.arange(9)
y = (np.arange(8) -0.2) * 1.1
```
Default parameters are exact Brunner Munzel test (two-sided)

```python
stat, pval, nb_iter_calc = permutation_test(x, y, test="brunner_munzel")
stat, pval, nb_iter_calc
# (-0.2776044311308564, 0.7475935828877005, 24310)
```

More on [usage.md](usage.md)

## Perspective 
Support for Wilcoxon's statistic should come soon.

## Cite
If you find this software useful for your work, please cite ... TBD.

## References
> [1] Brunner, E. and Munzel, U. (2000), The Nonparametric Behrens‐Fisher Problem:
> Asymptotic Theory and a Small‐Sample Approximation. Biom. J., 42: 17-25.
> doi:10.1002/(SICI)1521-4036(200001)42:1<17::AID-BIMJ17>3.0.CO;2-U

> [2] Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of
> Normality Implicit in the Analysis of Variance."
> Journal of the American Statistical Association 32(200): 675-701.
