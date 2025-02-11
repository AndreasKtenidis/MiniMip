from federator import NumpyAggregatorClient
from abc import ABC, abstractmethod
from typing import List, Union,Any,Callable, Tuple
from _agg_function import AggFunction
from _storage import StoredElement
from numpy import ndarray

# class RoundCall(ABC):
#     def __init__(self):
#         pass
#
#     @abstractmethod
#     def compute(self, *values)->List[Union[AggFunction,StoredElement]]:
#         pass
#

class FederatedAlgorithm(ABC):

    @abstractmethod
    def __init__(self,*round_calls:Callable[[NumpyAggregatorClient, ndarray], List[AggFunction]]):
        self.round_calls = round_calls

    def compute_round(self,alg_round:int,*args):
        self.round_calls[alg_round].compute(*args)

