import numpy as np
from federator import NumpyFedAggregator,NumpyClient



class Correlation:
    def __init__(self,aggregator:NumpyFedAggregator):
        self.x = np.random.random(10)
        self.y = np.random.random(10)
        self.aggregator=aggregator
        print(self.x)
        print(self.y)
        print(self.x*self.y)

    def __call__(self,message=None):
        print(np.sum(self.x*self.y))
        return self.aggregator.sum(self.x*self.y)

class LocalClient(NumpyClient):
    def global_sum(self,  vector) -> float:
        return vector

    def global_count(self, vector) -> int:
        return vector


f= Correlation(NumpyFedAggregator(LocalClient()))
print(f())






