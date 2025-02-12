from typing import List
from _abstract_algorithm import AggFunction
from _agg_function import AVG
from numpy import ndarray
from _client_server import LocalStorage
from _abstract_algorithm import FederatedAlgorithm

def func1(storage:LocalStorage, x:ndarray)-> List[AggFunction]:
    storage.store('x', x)
    return [AVG('avg_x',x**2)]

def func2(storage:LocalStorage, avg_x)-> List[AggFunction]:
    x=storage.load('x')
    return [AVG('dev_x',(x-avg_x)**2)]

algorithm   = FederatedAlgorithm(func1,func2)
