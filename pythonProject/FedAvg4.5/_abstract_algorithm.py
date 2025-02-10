from federator import NumpyAggregatorClient
from abc import ABC, abstractmethod
from typing import List, Type
from _agg_function import AggFunction

class RoundCall(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, *values)->tuple[List[AggFunction],int]:
        pass


class FederatedAlgorithm(ABC):
    def __init__(self,round_calls: List[RoundCall] ):
        self.round_calls = round_calls

    def compute_round(self,alg_round:int,*args):
        self.round_calls[alg_round].compute(*args)

