import numpy as np
from abc import ABC,abstractmethod

class NumpyClient(ABC):

    @abstractmethod
    def global_sum(self, local_sum)->float:
        pass

    @abstractmethod
    def global_count(self, local_count)->int:
        pass

class NumpyFedAggregator:
    def __init__(self, client:NumpyClient):
        self.client=client

    def sum(self, a):
        return self.client.global_sum(np.sum(a))

    def len(self, a):
        return self.client.global_count(np.count(a))

    def avg(self, a):
        return self.sum(a)/self.len(a)
