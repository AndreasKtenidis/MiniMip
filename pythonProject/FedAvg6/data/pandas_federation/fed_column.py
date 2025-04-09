import pandas as pd

from client.aggregation_client import NumpyAggregationClient


class ClientSeries(pd.Series):
    _metadata = ['client']

    def __init__(self, *args, client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    @property
    def _constructor(self):
        return ClientSeries

    def __repr__(self):
        base = super().__repr__()
        return f"ClientSeries(client={self.client})\n{base}"

    def fed_sum(self):
        return self.client.fed_sum(self.sum())

    def fed_count(self):
        return self.client.fed_count(self.count())

    def fed_avg(self):
        return self.client.fed_avg(self.sum(),self.count())

    def fed_min(self):
        return self.client.fed_min(self.min())

    def fed_max(self):
        return self.client.fed_max(self.max())

    def get_client(self)->NumpyAggregationClient:
        return self.client