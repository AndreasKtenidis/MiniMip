from scipy.stats import fisher_exact

from library.exaflow_ready.group_comparisons._cross_tab_table import CrossTabTable
from library.templates.statistical_function import StatisticalFunction


class FisherExact(StatisticalFunction):

    def compute(self, dataset, *, factor,factor_categories, outcome, outcome_categories):
        cross_tab_table = CrossTabTable(self.client).compute(dataset, factor=factor,
                                                             factor_categories=factor_categories,
                                                             outcome=outcome,
                                                             outcome_categories=outcome_categories)
        odds_ratio, p_value = fisher_exact(cross_tab_table)
        return odds_ratio, p_value
