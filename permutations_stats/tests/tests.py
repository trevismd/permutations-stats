from permutations_stats.tests import brunner_munzel, friedman, mann_whitney

TESTS = {"brunner_munzel": brunner_munzel.test,
         "friedman": friedman.test,
         "mann_whitney": mann_whitney.test}
