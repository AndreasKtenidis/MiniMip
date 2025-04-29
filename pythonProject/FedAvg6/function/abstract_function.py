from client.aggregation_client import AggregationClient, NumpyAggClient, PandasAggClient
from abc import ABC


class AggFunc(ABC):

    def __init__(self,client:AggregationClient):
        self.client = client

    def compute(self,*args, **kwargs):
        pass

    def get_numpy_aggregator(self):
        return NumpyAggClient(self.client)

    def get_pandas_aggregator(self):
        return PandasAggClient(self.client)

