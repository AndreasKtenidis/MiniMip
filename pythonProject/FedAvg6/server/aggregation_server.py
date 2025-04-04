from typing import List, Tuple

import numpy as np
from abc import ABC,abstractmethod
from itertools import chain




class NumpyAggregationServer(ABC):
    @staticmethod
    def sum(values:List[float]):
        return np.sum(np.array(values), axis=0).astype(np.float64)


    @staticmethod
    def min(values: List[float]):
        return np.min(np.array(values), axis=0).astype(np.float64)

    @staticmethod
    def max(values: List[float]):
        return np.max(np.array(values), axis=0).astype(np.float64)