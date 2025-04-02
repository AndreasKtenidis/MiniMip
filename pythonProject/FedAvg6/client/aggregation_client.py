import numpy as np
from abc import ABC,abstractmethod


class NumpyAggregationClient(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum):
        pass

    @abstractmethod
    def __global_min__(self, local_min):
        pass

    @abstractmethod
    def __global_max__(self, local_max):
        pass

    def sum(self, a):
        _ans = self.__global_sum__(np.stack([np.sum(np.asarray(a))], axis=0))
        return _ans[0]

    def count(self, a):
        _ans = self.__global_sum__(np.stack([len(np.asarray(a))], axis=0))
        return _ans[0]

    def avg(self, a):
        _ans = self.__global_sum__(np.stack([np.sum(np.asarray(a)), len(np.asarray(a))], axis=0))
        return _ans[0]/_ans[1]

    def min(self, a):
        _ans = self.__global_min__(np.stack([np.min(np.asarray(a))], axis=0))
        return _ans[0]

    def max(self, a):
        _ans = self.__global_max__(np.stack([np.max(np.asarray(a))], axis=0))
        return _ans[0]
