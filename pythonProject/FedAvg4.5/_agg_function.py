from federator import NumpyAggregatorClient
from abc import ABC, abstractmethod
from typing import List, Type
from numpy import ndarray
from constants import AGG
import numpy as np


class AggFunction(ABC):
    pass

class SUM(AggFunction):
    def __init__(self, x: ndarray):
        self.operation = AGG.SUM
        self.sum = np.sum(x)

class COUNT(AggFunction):
    def __init__(self,x:ndarray):
        self.operation = AGG.COUNT
        self.count = len(x)

class AVG(AggFunction):
    def __init__(self,x:ndarray):
        self.operation = AGG.SUM
        self.count = len(x)
        self.sum = np.sum(x)


