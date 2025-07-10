from system.client.aggregation_client import AggregationClient, NumpyAggClient, PandasAggClient
from abc import ABC, abstractmethod


class StatisticalModel(ABC):

    def __init__(self, client: AggregationClient):
        self.client = client

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def get_numpy_aggregator(self) -> NumpyAggClient:
        return NumpyAggClient(self.client)

    def get_pandas_aggregator(self)->PandasAggClient:
        return PandasAggClient(self.client)
