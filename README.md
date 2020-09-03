![coverage](.github/coverage.svg)
# permutations-stats
Python only permutation-based statistical tests, accelerated with numba.
## Status
Brunner Munzel and Friedman tests (repeated measures) are implemented at the moment.   
Exact testing (all permutations) and approximate method (simulation) are available.

In the current implementation, a few seconds are required to compile on the fly.
Then, acceleration is critical as shown in this output from tests comparing the
statistic calculation with scipy:

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
## Dependencies
* numpy
* numba

And for development testing only
* scipy, pytest

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
stat, pval, nb_iter_calc = permutation_test(x, y, test="brunner_munzel")
stat, pval, nb_iter_calc
# (-0.2776044311308564, 0.7475935828877005, 24310)
```

Simulations are run like this (n_iter has default 10 000 iterations and can be omitted)
NB: # 1 is added to numerator and denominator of pvalue calculation if simulations are run
```python
stat, pval, nb_iter_calc = permutation_test(x, y, method="approximate", n_iter=100)
stat, pval, nb_iter_calc
# (-0.2776044311308564, 0.7227722772277227, 101)  
```

An exact test is run if the number of iterations specified is larger than number of combinations
```python
stat, pval, nb_iter_calc = permutation_test(x, y, method="approximate", n_iter=100_000)
# Simulation overridden by exact test because total number of combinations (24310)
# is smaller than asked amount of simulation iterations (100000).
# Pass `force_simulations=True` to avoid this behavior

stat, pval, nb_iter_calc
# (-0.2776044311308564, 0.7475935828877005, 24310)
```

Other alternatives are possible (`"greater"` and `"less"`, default is `"two-sided"`)
```python
_, pval, _ = permutation_test(x, y, alternative="greater")
pval
# 0.37379679144385025
```

## Cite
If you find this software useful for your work, please cite ...

## References
> Brunner, E. and Munzel, U. (2000), The Nonparametric Behrens‐Fisher Problem:
> Asymptotic Theory and a Small‐Sample Approximation. Biom. J., 42: 17-25.
> doi:10.1002/(SICI)1521-4036(200001)42:1<17::AID-BIMJ17>3.0.CO;2-U

> Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of
> Normality Implicit in the Analysis of Variance."
> Journal of the American Statistical Association 32(200): 675-701.
