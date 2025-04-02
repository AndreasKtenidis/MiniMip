from client.aggregation_client import NumpyAggregationClient
from abc import ABC


class AggFunc(ABC):


    def compute(self,*args, **kwargs):
        pass


