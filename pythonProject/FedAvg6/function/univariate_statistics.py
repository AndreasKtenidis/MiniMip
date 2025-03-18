from function.abstract_function import AggFunc
import math

class Variance(AggFunc):
    def compute(self, x):
        _count = self.count(x)
        if self.count(x)<=1:
            return 0
        sum_of_squares = self.avg(x ** 2)
        return sum_of_squares - self.avg(x) ** 2

class StandardDeviation(AggFunc):
    def compute(self, x):
        return math.sqrt(Variance(self.agg_client).compute(x))

class SumOfSquares(AggFunc):
    def compute(self, x):
        return self.sum(x ** 2)

class Range(AggFunc):
    def compute(self, x):
        return self.max(x) - self.min(x)

class CoefficientOfVariation(AggFunc):
    def compute(self, x):
        avg_data = self.avg(x)
        if avg_data == 0:
            return 0
        StandardDeviation(self.agg_client).compute(x)/avg_data

class MeanAbsoluteDeviation(AggFunc):
    def compute(self, x):
        if self.count(x) == 0:
            return 0
        avg_data = self.avg(x)
        return self.avg(abs(x - avg_data))

class RootMeanSquare(AggFunc):
    def compute(self, x):
        if self.count(x) == 0:
            return 0
        return math.sqrt(self.avg(x ** 2))

class MeanSquare(AggFunc):
    def compute(self, x):
        y = self.agg_client.avg(x)
        z=y-x
        return self.agg_client.avg(z**2)