import numpy as np
from abc import ABC,abstractmethod

from pyexpat.errors import messages


class NumpyClient(ABC):

    @abstractmethod
    def global_sum(self,message, local_sum)->float:
        pass

    @abstractmethod
    def global_count(self, message,local_count)->int:
        pass

class NumpyFedAggregator:
    def __init__(self, client:NumpyClient, message):
        self.message = message
        self.client=client

    def sum(self, a):
        return self.client.global_sum(self.message,np.sum(a))

    def len(self, a):
        return self.client.global_count(self.message,np.count(a))

    def avg(self, a):
        return self.sum(a)/self.len(a)
