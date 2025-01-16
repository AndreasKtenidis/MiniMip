from abc import ABC, abstractmethod
import math

from server_flower import FlowerExecutor
from typing import Dict


class Complex_Operation():

    def __init__(self, executor:FlowerExecutor):
        self.executor = executor

    def value(self)->float:

        mx = self.executor.AVG('x')
        my = self.executor.AVG('y')
        mxy = self.executor.AVG('x*y')
        sx = math.sqrt(self.executor.AVG('x**2') - mx ** 2)
        sy = math.sqrt(self.executor.AVG('y**2',) - my ** 2)
        return (mxy - mx * my) / (sx * sy)

