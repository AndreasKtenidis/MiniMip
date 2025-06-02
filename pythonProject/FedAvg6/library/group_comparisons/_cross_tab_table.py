from library.stats.statistical_function import StatisticalFunction
import pandas as pd
from scipy.stats import chi2_contingency


class CrossTabTable(StatisticalFunction):

    def compute(self, factor_to_outcome):
        agg = self.get_numpy_aggregator()
        df = pd.DataFrame(factor_to_outcome, columns=['factor', 'outcome'])
        cross_tab = pd.crosstab(df['factor'], df['outcome'])
        rows_ = cross_tab.index.values
        columns_ = cross_tab.columns.values
        rows = rows_
        columns = columns_

        rows = agg.fed_union(rows)
        columns = agg.fed_union(columns)

        for row in set(rows) - set(rows_):
            new_row = pd.Series(0, index=cross_tab.columns, name=row)
            cross_tab = pd.concat([cross_tab, pd.DataFrame([new_row])])
        for column in set(columns) - set(columns_):
            cross_tab[column] = 0

        cross_tab = cross_tab.reindex(columns=columns, index=rows)

        cross_tab.iloc[:] = agg.fed_sum(cross_tab.values)
        return cross_tab