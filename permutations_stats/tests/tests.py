from permutations_stats.tests import brunner_munzel, friedman, mann_whitney, \
    wilcoxon

TESTS = {"brunner_munzel": brunner_munzel.test,
         "friedman": friedman.test,
         "mann_whitney": mann_whitney.test,
         "wilcoxon": wilcoxon.test}
