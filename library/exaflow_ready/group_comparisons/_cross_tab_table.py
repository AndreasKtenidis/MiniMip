
import pandas as pd

from library.templates.statistical_function import StatisticalFunction


class CrossTabTable(StatisticalFunction):

    def compute(self, dataset, *, factor,factor_categories, outcome, outcome_categories):
        # Convert columns to categorical with the full set of categories
        dataset[factor] = pd.Categorical(dataset[factor], categories=factor_categories)
        dataset[outcome] = pd.Categorical(dataset[outcome], categories=outcome_categories)
        agg = self.get_numpy_aggregator()
        cross_tab = pd.crosstab(dataset[factor],
                                dataset[outcome],
                                dropna=False)
        cross_tab.iloc[:] = agg.fed_sum(cross_tab.values)
        return cross_tab
