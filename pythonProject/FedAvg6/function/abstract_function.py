from client.aggregation_client import NumpyAggregationClient
from abc import ABC


class AggFunc(ABC):
    def __init__(self,agg_client: NumpyAggregationClient ):
        self.agg_client = agg_client

    def sum(self,x):
        return self.agg_client.sum(x)

    def avg(self,x):
        return self.agg_client.avg(x)

    def count(self,x):
        return self.agg_client.count(x)

    def min(self,x):
        return self.agg_client.min(x)

    def max(self,x):
        return self.agg_client.max(x)

    def compute(self,*args, **kwargs):
        pass


