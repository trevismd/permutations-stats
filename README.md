![coverage](.github/coverage.svg)
[![Active Development](https://img.shields.io/badge/Maintenance%20Level-Actively%20Developed-brightgreen.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
# permutations-stats
Python-only permutation-based statistical tests, accelerated with numba.
## Status
### Statistical tests
Brunner Munzel [1], Mann Whitney Wilcoxon [2, 3], Wilcoxon signed rank test [3],
and Friedman [4] tests are implemented.

Exact tests (all combinations) and approximate method (simulation) are available.

Functions for bootstrapping-based confidence intervals calculations for
mean, median and standard deviation are also implemented (not documented yet).

## Why this package ?
This work aims to provide fast permutation-based statistical tests in Python.
Some tests are not available publicly in an exact mode (computing all
possible permutations) or with simulations. If certain assumptions cannot be
made about the data (such as normality) or if the sample is not large enough, the
existing implementations should not be used.  
For example, the Brunner Munzel [1] test is implemented in `scipy` but not with
an exact calculation. The statistic can be used with the public API but it can
take some time if ran several thousands of times (e.g. the p-value is also 
calculated for each iteration).  

This packages reimplements the looping of permutations and statistical tests
with `numba`. A few seconds are required to compile on the fly for the first 
function call. Then, acceleration is critical as shown in this output from 
tests comparing the Brunner Munzel statistic calculation with `scipy`:

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
* [numpy](https://www.numpy.org/)
* [numba](https://numba.pydata.org/)

And for development testing only
* [scipy](https://www.scipy.org/) &geq;1.5
* [pytest](https://www.pytest.org/)

## Usage
Basic usage:
```python
import numpy as np
from permutations_stats.permutations import permutation_test

# Sample data
x = np.arange(9)
y = (np.arange(9) -0.2) * 1.1

permutation_test(x, y, test="brunner_munzel")
# PermutationsResults(statistic=0.2776044311308564, pvalue=0.7475935828877005, permutations=24310, test='brunner_munzel', alternative='TWO_SIDED', method='exact')
```

More examples on [usage.ipynb](usage.ipynb) and a detailed demonstration on 
[doc/demo.ipynb](doc/demo.ipynb).

## Perspective
* If sample sizes and/or the number of iterations are small, acceleration is
not expected with `numba`, and using `numpy` alone should be the fastest option.  
Thresholds for `numba` use will be better determined to decide the function to call
(without user intervention).

## License
GNU General Public License v3.0 only.

## Cite
If you find this software useful for your academic work, please cite ... TBD.

## Acknowledgements
We would like to thank Marianne Paesmans, Lieveke Ameye and Luigi Moretti at 
Institut Jules Bordet for their support during the development of this package.  

## References
> [1] Brunner, E. and Munzel, U. (2000), The Nonparametric Behrens‐Fisher
> Problem: Asymptotic Theory and a Small‐Sample Approximation. Biom. J., 42:
> 17-25. doi:10.1002/(SICI)1521-4036(200001)42:1<17::AID-BIMJ17>3.0.CO;2-U

> [2] Mann, H. B. and D. R. Whitney (1947). "On a Test of Whether one of Two
> Random Variables is Stochastically Larger than the Other." Ann. Math. Statist.
> 18(1): 50-60.

> [3] Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods."
> Biometrics Bulletin 1(6): 80-83.

> [4] Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of
> Normality Implicit in the Analysis of Variance."
> Journal of the American Statistical Association 32(200): 675-701.
