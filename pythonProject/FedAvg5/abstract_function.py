from federator import NumpyAggregatorClient
from abc import ABC, abstractmethod


class AggFunc(ABC):
    def __init__(self,agg_client: NumpyAggregatorClient ):
        self.agg_client = agg_client

    def compute(self,*args, **kwargs):
        pass

class Avg_Power(AggFunc):
    def compute(self, x):
        y = x ** 2
        return self.agg_client.sum(y)
