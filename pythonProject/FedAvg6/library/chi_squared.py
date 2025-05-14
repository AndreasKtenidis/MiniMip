import pandas as pd
import numpy as np
from urllib.request import urlopen

from function.abstract_function import AggFunc
import pandas as pd
import random
from scipy.stats import chi2_contingency


class ChiSquared(AggFunc):

    def compute(self, factor_to_outcome):
        # TODO: Currently works only if the categories are numerical
        agg = self.get_numpy_aggregator()
        df = pd.DataFrame(factor_to_outcome, columns=['factor', 'outcome'])
        contingency_table =pd.crosstab(df['factor'], df['outcome'])
        rows_ = contingency_table.index.values
        columns_ = contingency_table.columns.values
        rows = rows_
        columns = columns_

        rows=agg.fed_union(rows)
        columns = agg.fed_union(columns)

        for row in set(rows)-set(rows_):
            new_row = pd.Series(0, index=contingency_table.columns, name=row)
            contingency_table = pd.concat([contingency_table, pd.DataFrame([new_row])])


        for column in set(columns)-set(columns_):
            contingency_table[column] = 0
        contingency_table=contingency_table.reindex(columns=columns,index=rows)

        contingency_table.iloc[:] = agg.fed_sum(contingency_table.values)

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p, dof, expected