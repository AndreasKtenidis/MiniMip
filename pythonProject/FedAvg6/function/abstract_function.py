from client.aggregation_client import AggregationClient
from abc import ABC


class AggFunc(ABC):


    def compute(self,*args, **kwargs):
        pass


