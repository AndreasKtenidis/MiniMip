from typing import List, Tuple

import numpy as np
from abc import ABC,abstractmethod
from itertools import chain




class NumpyAggregationServer(ABC):
    @staticmethod
    def sum(values:List):
        return np.sum(np.array(values), axis=0).astype(np.float64)


    @staticmethod
    def min(values: List):
        return np.min(np.array(values), axis=0).astype(np.float64)

    @staticmethod
    def max(values: List):
        return np.max(np.array(values), axis=0).astype(np.float64)

    @staticmethod
    def union(values: List):
        print('-->',values)
        union_ = list(set(chain.from_iterable(values)))
        print(union_)
        return union_
