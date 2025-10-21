from library.utils.aggregation_client import AggregationClientInterface
from abc import ABC, abstractmethod


class StatisticalFunction(ABC):

    def __init__(self, client: AggregationClientInterface):
        self.client = client

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass
