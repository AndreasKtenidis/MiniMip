import numpy as np
from abc import ABC,abstractmethod

from pyexpat.errors import messages


class NumpyClient(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum)->float:
        pass

    @abstractmethod
    def __global_count__(self, local_count)->int:
        pass

    def sum(self, a):
        return self.__global_sum__(np.sum(a))

    def count(self, a):
        return self.__global_count__(len(a))

    def avg(self, a):
        return self.sum(a)/self.count(a)

