from typing import List, Union, Any, Callable
from _abstract_algorithm import AggFunction
from _agg_function import SUM,COUNT,AVG
from numpy import ndarray
from _client_server import NumpyAggregatorClient


def func1(client:NumpyAggregatorClient,x:ndarray)-> List[AggFunction]:
    client.store('x',x)
    return [AVG('avg_x',x**2)]

def func2(client:NumpyAggregatorClient,y:ndarray)-> List[AggFunction]:
    x=client.load('x')
    return [AVG('dev_x',x**2-y)]

algorithmic_steps  = [func1,func2]
