from data.fed_table import FedTable
from function.abstract_function import AggFunc
import math
import numpy as np

class Dummy(AggFunc):
    def compute(self, x):
        random_integers = np.random.randint(1, 11, size=3)
        print(random_integers)
        return 3

class Variance(AggFunc):
    def compute(self, x:FedTable):
        _count = x.fed_count()
        if _count<=1:
            return 0
        sum_of_squares = (x ** 2).fed_avg()
        return sum_of_squares -(x .fed_avg()** 2)

class StandardDeviation(AggFunc):
    def compute(self, x:FedTable):
        return math.sqrt(Variance().compute(x))

class SumOfSquares(AggFunc):
    def compute(self, x:FedTable):
        return (x ** 2).sum()

class Range(AggFunc):
    def compute(self, x:FedTable):
        return x.max() - x.min()

class CoefficientOfVariation(AggFunc):
    def compute(self, x:FedTable):
        avg_data = x.fed_avg()
        if avg_data == 0:
            return 0
        StandardDeviation().compute(x)/avg_data

class MeanAbsoluteDeviation(AggFunc):
    def compute(self, x:FedTable):
        if x.fed_count() == 0:
            return 0
        avg_data = x.fed_avg()
        return abs(x - avg_data).fed_avg()

class RootMeanSquare(AggFunc):
    def compute(self, x:FedTable):
        if x.fed_count() == 0:
            return 0
        return math.sqrt((x ** 2).fed_avg())

class MeanSquare(AggFunc):
    def compute(self, x:FedTable):
        y = x.fed_avg()
        z=y-x
        return (z**2).fed_avg()