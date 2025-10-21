from abc import ABC

import numpy as np

from library.utils.aggregation_client import AggregationClientInterface


class GrizzlyAggClient():
    """A client for performing federated operations on Grizzly DataFrames.

    This class provides an interface for common federated aggregation operations
    that communicate with a central aggregation server through an AggregationClient.
    """

    def __init__(self, client: AggregationClientInterface):
        self.client = client

    def global_sum(self, dataframe):
        _agg = dataframe.sum()
        _ans = self.client.__global_sum__([_agg])
        return _ans

    def global_avg(self, dataframe):
        _agg = dataframe.mean()
        means = [row[1] for row in _agg.collect()]
        counts_df = dataframe.count()
        counts = [row[1] for row in counts_df.collect()]
        means_with_count = means + counts
        _ans = self.client.__global_sum__(means_with_count)
        _ans = np.array(_ans[0:len(_ans) // 2]) / _ans[len(_ans) // 2:]
        return _ans

    def global_count(self, dataframe):
        local_count = dataframe.count()
        total_count = self.client.__global_sum__([local_count])
        return total_count[0]
