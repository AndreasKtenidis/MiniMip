import numpy as np
from abc import ABC,abstractmethod


class NumpyAggregationClient(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum):
        pass

    @abstractmethod
    def __global_count__(self, local_count):
        pass

    @abstractmethod
    def __global_avg__(self, local_sum, local_count):
        pass

    def sum(self, a):
        return self.__global_sum__(np.sum(np.asarray(a)))

    def count(self, a):
        return self.__global_count__(len(np.asarray(a)))

    def avg(self, a):
        print('---->',a)
        print('---->',np.sum(a))
        return self.__global_avg__(np.sum(np.asarray(a)),len(np.asarray(a)))

