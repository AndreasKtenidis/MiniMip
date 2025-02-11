from typing import List, Union,Any
from _abstract_algorithm import AggFunction
from _agg_function import SUM,COUNT,AVG
from _abstract_algorithm import FederatedAlgorithm,  StoredElement
from numpy import ndarray
from federator import NumpyAggregatorClient


def func1(client:NumpyAggregatorClient,x:ndarray)-> List[AggFunction]:
    client.store('x',x)
    return [SUM(x**2)]

def func2(client:NumpyAggregatorClient,y:ndarray)-> List[AggFunction]:
    x=client.load('x')
    return [SUM(x**2-y)]
