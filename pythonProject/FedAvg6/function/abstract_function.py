from client.aggregation_client import NumpyAggregationClient
from abc import ABC


class AggFunc(ABC):
    def __init__(self,agg_client: NumpyAggregationClient ):
        self.agg_client = agg_client

    def compute(self,*args, **kwargs):
        pass

class MeanSquare(AggFunc):
    def compute(self, x):
        y = self.agg_client.avg(x)
        z=y-x
        return self.agg_client.avg(z**2)
