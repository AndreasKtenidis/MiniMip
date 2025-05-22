from client.aggregation_client import AggregationClient, NumpyAggClient, PandasAggClient
from abc import ABC


class AggTransformer(ABC):

    def __init__(self,client:AggregationClient):
        self.client = client



    def get_numpy_aggregator(self):
        return NumpyAggClient(self.client)

    def get_pandas_aggregator(self):
        return PandasAggClient(self.client)

