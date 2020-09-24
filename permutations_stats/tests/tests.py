import permutations_stats.utils as pmu
from permutations_stats.tests import brunner_munzel, friedman, mann_whitney, \
    wilcoxon

TESTS = {"brunner_munzel": {
            pmu.Alternative.TWO_SIDED: {
                "first": brunner_munzel.abs_test,
                "then": brunner_munzel.abs_test},
            pmu.Alternative.GREATER: {
                "first": brunner_munzel.invert_test,
                "then": brunner_munzel.invert_test},
            pmu.Alternative.LESS: {
                "first": brunner_munzel.invert_test,
                "then": brunner_munzel.invert_test}},

         "friedman": {
            pmu.Alternative.TWO_SIDED: {
                "first": friedman.test,
                "then": friedman.test},
            pmu.Alternative.GREATER: {
                "first": friedman.test,
                "then": friedman.test},
            pmu.Alternative.LESS: {
                "first": friedman.test,
                "then": friedman.test}},

         "friedman_faster": {
            pmu.Alternative.TWO_SIDED: {
                "first": friedman.test_faster,
                "then": friedman.test_faster}},

         "mann_whitney": {
            pmu.Alternative.TWO_SIDED: {
                "first": mann_whitney.max_u,
                "then": mann_whitney.max_u},
            pmu.Alternative.GREATER: {
                "first": mann_whitney.test,
                "then": mann_whitney.test},
            pmu.Alternative.LESS: {
                "first": mann_whitney.test,
                "then": mann_whitney.test}},

         "wilcoxon": {
            pmu.Alternative.TWO_SIDED: {
                "first": wilcoxon.abs_test,
                "then": wilcoxon.abs_test},
            pmu.Alternative.GREATER: {
                "first": wilcoxon.test,
                "then": wilcoxon.test},
            pmu.Alternative.LESS: {
                "first": wilcoxon.test,
                "then": wilcoxon.test}}
        }
