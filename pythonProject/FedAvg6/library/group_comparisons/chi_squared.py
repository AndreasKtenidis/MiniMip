from library.group_comparisons._cross_tab_table import CrossTabTable
from library.templates.statistical_function import StatisticalFunction
from scipy.stats import chi2_contingency


class ChiSquared(StatisticalFunction):

    def compute(self, dataset,*,factor,outcome):
        cross_tab_table = CrossTabTable(self.client).compute(dataset,factor=factor,outcome=outcome)
        chi2, p, dof, expected = chi2_contingency(cross_tab_table)
        return chi2, p, dof, expected