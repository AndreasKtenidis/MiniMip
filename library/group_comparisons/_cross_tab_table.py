
import pandas as pd

from library.templates.statistical_function import StatisticalFunction


class CrossTabTable(StatisticalFunction):

    def compute(self, dataset, *, factor, outcome):
        agg = self.get_numpy_aggregator()
        df = pd.DataFrame(dataset, columns=[factor, outcome])
        cross_tab = pd.crosstab(df[factor], df[outcome])
        rows_ = cross_tab.index.values
        columns_ = cross_tab.columns.values
        rows = rows_
        columns = columns_

        for row in set(rows) - set(rows_):
            new_row = pd.Series(0, index=cross_tab.columns, name=row)
            cross_tab = pd.concat([cross_tab, pd.DataFrame([new_row])])
        for column in set(columns) - set(columns_):
            cross_tab[column] = 0

        cross_tab = cross_tab.reindex(columns=columns, index=rows)

        cross_tab.iloc[:] = agg.fed_sum(cross_tab.values)

        return cross_tab
