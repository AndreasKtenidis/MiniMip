from client.aggregation_client import NumpyAggregationClient
from abc import ABC


class AggFunc(ABC):
    def __init__(self,agg_client: NumpyAggregationClient ):
        self.agg_client = agg_client

    def compute(self,*args, **kwargs):
        pass


