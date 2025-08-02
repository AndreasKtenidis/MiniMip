from scipy.stats import fisher_exact

from library.group_comparisons._cross_tab_table import CrossTabTable
from library.templates.statistical_function import StatisticalFunction


class FisherExact(StatisticalFunction):

    def compute(self, dataset, *, factor, outcome):
        cross_tab_table = CrossTabTable(self.client).compute(dataset, factor=factor, outcome=outcome)
        odds_ratio, p_value = fisher_exact(cross_tab_table)
        return odds_ratio, p_value
