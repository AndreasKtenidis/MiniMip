import numpy as np
from abc import ABC,abstractmethod


class AggregationClient(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum):
        pass

    @abstractmethod
    def __global_min__(self, local_min):
        pass

    @abstractmethod
    def __global_max__(self, local_max):
        pass

    def fed__avg(self,array:np.ndarray):
        tmp = np.append(array, 1)
        tmp = self.__global_sum__( tmp)
        return np.array(tmp[:-1])/tmp[-1]

class NumpyAggClient(AggregationClient, ABC):
    pass

class PandasAggClient(AggregationClient, ABC):
    pass