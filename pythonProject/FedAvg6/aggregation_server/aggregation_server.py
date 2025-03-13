from typing import List, Tuple

import numpy as np
from abc import ABC,abstractmethod




class NumpyAggregationServer(ABC):
    @staticmethod
    def sum(values:List[float]):
        return sum(values)

    @staticmethod
    def count(values: List[float]):
        return sum(values)

    @staticmethod
    def avg(values: List[Tuple[float,float]]):
        _avg, _count = map(sum, zip(*values))
        return _avg /_count
