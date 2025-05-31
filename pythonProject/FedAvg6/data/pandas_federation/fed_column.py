from system.client.aggregation_client import AggregationClient


import pandas as pd

class FedSeries(pd.Series):
    # Declare custom attributes in _metadata
    _metadata = ['client']

    def __init__(self, *args, client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client  # will now be handled safely by pandas

    @property
    def _constructor(self):
        return FedSeries

    def fed_sum(self):
        _agg = ([self.sum()])
        _ans = self.client.__global_sum__(_agg)
        return _ans[0]

    def fed_count(self):
        _agg = ([self.count()])
        _ans = self.client.__global_sum__(_agg)
        return _ans[0]


    def fed_min(self):
        _agg = ([self.min()])
        _ans = self.client.__global_min__(_agg)
        return _ans[0]

    def fed_max(self):
        _agg = ([self.max()])
        _ans = self.client.__global_max__(_agg)
        return _ans[0]

    def fed_avg(self):
        _sum = (self.sum())
        _count = (self.count())
        _ans = self.client.__global_sum__([_sum, _count])
        return _ans[0] / _ans[1]

    def get_client(self) -> AggregationClient:
        return self.client