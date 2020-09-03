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
