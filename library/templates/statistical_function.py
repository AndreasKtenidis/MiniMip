from mini_mip_system.client.aggregation_client import AggregationClient, NumpyAggClient, PandasAggClient
from abc import ABC, abstractmethod


class StatisticalFunction(ABC):

    def __init__(self, client: AggregationClient):
        self.client = client

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def get_numpy_aggregator(self) -> NumpyAggClient:
        return NumpyAggClient(self.client)

    def get_pandas_aggregator(self):
        return PandasAggClient(self.client)

    def get_grizzly_aggregator(self):
        return GrizzlyAggClient(self.client)
