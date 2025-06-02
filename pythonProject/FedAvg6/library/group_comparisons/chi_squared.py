from library.group_comparisons._cross_tab_table import CrossTabTable
from library.stats.statistical_function import StatisticalFunction
import pandas as pd
from scipy.stats import chi2_contingency


class ChiSquared(StatisticalFunction):

    def compute(self, factor_to_outcome):
        cross_tab_table = CrossTabTable(self.client).compute(factor_to_outcome)
        chi2, p, dof, expected = chi2_contingency(cross_tab_table)
        return chi2, p, dof, expected