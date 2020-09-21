import permutations_stats.utils as pmu
from permutations_stats.tests import brunner_munzel, friedman, mann_whitney, \
    wilcoxon

TESTS = {"brunner_munzel": {pmu.Alternative.TWO_SIDED: brunner_munzel.abs_test,
                            pmu.Alternative.GREATER: brunner_munzel.test,
                            pmu.Alternative.LESS: brunner_munzel.test},
         "friedman": {pmu.Alternative.TWO_SIDED: friedman.test,
                      pmu.Alternative.GREATER: friedman.test,
                      pmu.Alternative.LESS: friedman.test},
         "mann_whitney": {pmu.Alternative.TWO_SIDED: mann_whitney.max_u,
                          pmu.Alternative.GREATER: mann_whitney.test,
                          pmu.Alternative.LESS: mann_whitney.test},
         "wilcoxon": {pmu.Alternative.TWO_SIDED: wilcoxon.abs_test,
                      pmu.Alternative.GREATER: wilcoxon.test,
                      pmu.Alternative.LESS: wilcoxon.test}
         }
