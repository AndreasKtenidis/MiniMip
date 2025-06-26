from scipy.stats import fisher_exact

from pythonProject.FedAvg6.library.group_comparisons._cross_tab_table import CrossTabTable
from pythonProject.FedAvg6.library.templates.statistical_function import StatisticalFunction

class FisherExact(StatisticalFunction):

    def compute(self, factor_to_outcome):
        cross_tab_table = CrossTabTable(self.client).compute(factor_to_outcome)
        odds_ratio, p_value = fisher_exact(cross_tab_table)
        return odds_ratio, p_value