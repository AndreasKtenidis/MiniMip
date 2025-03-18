from typing import List, Tuple

import numpy as np
from abc import ABC,abstractmethod
from itertools import chain




class NumpyAggregationServer(ABC):
    @staticmethod
    def sum(values:List[float]):
        _sum = sum(chain(*values))
        return _sum

    @staticmethod
    def count(values: List[float]):
        _count = sum(chain(*values))
        return _count

    @staticmethod
    def avg(values: List[Tuple[float,float]]):
        _sum, _count = map(sum, zip(*values))
        return _sum /_count

    @staticmethod
    def min(values: List[float]):
        return min(chain(*values))

    @staticmethod
    def max(values: List[float]):
        return max(chain(*values))