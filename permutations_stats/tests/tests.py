import permutations_stats.utils as pmu
from permutations_stats.tests import brunner_munzel, friedman, mann_whitney, \
    wilcoxon

# noinspection PyProtectedMember
TESTS = {"brunner_munzel": {
            pmu.Alternative.TWO_SIDED: {
                "first": brunner_munzel.abs_test,
                "then": brunner_munzel.abs_test_faster},
            pmu.Alternative.GREATER: {
                "first": brunner_munzel.invert_test,
                "then": brunner_munzel.invert_test_faster},
            pmu.Alternative.LESS: {
                "first": brunner_munzel.invert_test,
                "then": brunner_munzel.invert_test_faster}},

         "friedman": {
            pmu.Alternative.TWO_SIDED: {
                "transform": friedman.transform,
                "first": friedman._test,
                "then": friedman.test_faster},
            pmu.Alternative.GREATER: {
                "transform": friedman.transform,
                "first": friedman._test,
                "then": friedman.test_faster},
            pmu.Alternative.LESS: {
                "transform": friedman.transform,
                "first": friedman._test,
                "then": friedman.test_faster}},

         "friedman_t2": {
            pmu.Alternative.TWO_SIDED: {
                "transform": friedman.transform,
                "first": friedman._test_t2,
                "then": friedman.test_faster}},

         "mann_whitney": {
            pmu.Alternative.TWO_SIDED: {
                "first": mann_whitney.test,
                "then": mann_whitney.max_u_faster},
            pmu.Alternative.GREATER: {
                "first": mann_whitney.test,
                "then": mann_whitney.test_faster},
            pmu.Alternative.LESS: {
                "first": mann_whitney.test,
                "then": mann_whitney.test_faster}},

         "wilcoxon": {
            pmu.Alternative.TWO_SIDED: {
                "first": wilcoxon.abs_test,
                "then": wilcoxon.abs_test_faster},
            pmu.Alternative.GREATER: {
                "first": wilcoxon._test,
                "then": wilcoxon._test_faster},
            pmu.Alternative.LESS: {
                "first": wilcoxon._test,
                "then": wilcoxon._test_faster}}
         }
