
from abc import ABC

from numpy import ndarray
from _constants import AGG
import numpy as np


class AggFunction(ABC):
    def __init__(self, var_name:str, operation:AGG, local_aggregations):
        self.operation = operation
        self.local_aggregations = local_aggregations
        self.var_name= var_name

    def get_operation(self)->AGG:
        return self.operation

    def get_local_aggregations(self):
        return self.local_aggregations

    def get_var_name(self):
        return self.var_name

class SUM(AggFunction):
    def __init__(self, var_name:str, x: ndarray):
        super().__init__(var_name, AGG.SUM,np.sum(x))

class COUNT(AggFunction):
    def __init__(self,var_name:str, x:ndarray):
        super().__init__(var_name, AGG.COUNT, len(x))

class AVG(AggFunction):
    def __init__(self,var_name:str, x:ndarray):
        super().__init__(var_name, AGG.AVG, [float(np.sum(x)),float(len(x))] )
