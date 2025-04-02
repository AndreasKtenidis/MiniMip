import numpy as np
from abc import ABC,abstractmethod


class NumpyAggregationClient(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum):
        pass

    # @abstractmethod
    # def __global_count__(self, local_count):
    #     pass

    # @abstractmethod
    # def __global_avg__(self, local_sum, local_count):
    #     pass

    @abstractmethod
    def __global_min__(self, local_min):
        pass

    @abstractmethod
    def __global_max__(self, local_max):
        pass

    def sum(self, a):
        _ans = self.__global_sum__(np.sum(np.asarray(a)))
        return _ans.answer

    def count(self, a):
        return self.__global_sum__(len(np.asarray(a)))

    def avg(self, a):
        _ans = self.__global_sum__(np.stack((np.sum(np.asarray(a)), len(np.asarray(a))), axis=0))
        return _ans.answer[0]/_ans.answer[1]

    def min(self, a):
        _ans = self.__global_min__(np.min(np.asarray(a)))
        return _ans.answer

    def max(self, a):
        _ans = self.__global_max__(np.max(np.asarray(a)))
        return _ans.answer
