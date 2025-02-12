from _client_server import LocalStorage
from abc import ABC
from typing import List, Callable
from _agg_function import AggFunction
from numpy import ndarray

class FederatedAlgorithm(ABC):


    def __init__(self,*round_calls:Callable[[LocalStorage, ndarray], List[AggFunction]]):
        self.round_calls = round_calls

    def get_operation(self,alg_round:int):
        return self.round_calls[alg_round]

