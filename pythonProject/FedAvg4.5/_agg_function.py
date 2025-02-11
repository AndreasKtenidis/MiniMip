
from abc import ABC, abstractmethod
from numpy import ndarray
from constants import AGG
import numpy as np


class AggFunction(ABC):
    def __init__(self,operation, local_aggregations: dict[str, int]):
        self.operation = operation
        self.local_aggregations = local_aggregations
    def get_operation(self):
        return self.operation

    def get_local_aggregations(self):
        return self.local_aggregations



class SUM(AggFunction):
    def __init__(self, x: ndarray):
        super().__init__(AGG.SUM,{AGG.SUM.value:np.sum(x)})


class COUNT(AggFunction):
    def __init__(self,x:ndarray):
        super().__init__(AGG.COUNT, {AGG.COUNT.value: len(x)})

class AVG(AggFunction):
    def __init__(self,x:ndarray):
        super().__init__(AGG.AVG, {AGG.SUM.value:np.sum(x), AGG.COUNT.value: len(x)})

