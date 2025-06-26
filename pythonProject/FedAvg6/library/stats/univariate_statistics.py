from pythonProject.FedAvg6.library.templates.statistical_function import StatisticalFunction
import numpy as np

class Dummy(StatisticalFunction):
    def compute(self, x):
        random_integers = np.random.randint(1, 11, size=3)
        print(random_integers)
        return 3

class Variance(StatisticalFunction):
    def compute(self, x:np.array):
        agg = self.get_numpy_aggregator()
        _count = agg.global_count(x)
        if _count<=1:
            return 0
        sum_of_squares = agg.global_avg(x ** 2)
        return sum_of_squares -(agg.global_avg(x)** 2)

class StandardDeviation(StatisticalFunction):
    def compute(self, x:np.array):
        return np.sqrt(Variance(self.client).compute(x))

class SumOfSquares(StatisticalFunction):
    def compute(self, x:np.array):
        agg = self.get_numpy_aggregator()
        return agg.global_sum(x ** 2)

class Range(StatisticalFunction):
    def compute(self, x:np.array):
        agg = self.get_numpy_aggregator()
        return agg.global_max( x) - agg.global_min( x)

class CoefficientOfVariation(StatisticalFunction):
    def compute(self, x:np.array):
        agg = self.get_numpy_aggregator()
        avg_data = agg.global_avg(x)
        if avg_data == 0:
            return 0
        StandardDeviation(self.client).compute(x)/avg_data

class MeanAbsoluteDeviation(StatisticalFunction):
    def compute(self, x:np.array):
        agg = self.get_numpy_aggregator()
        if agg.global_count(x)== 0:
            return 0
        avg_data = agg.global_avg(x)
        return agg.global_avg(abs(x - avg_data))

class RootMeanSquare(StatisticalFunction):
    def compute(self, x:np.array):
        agg = self.get_numpy_aggregator()
        if agg.global_count(x)== 0:
            return 0
        return agg.global_avg(np.sqrt(x ** 2))

class MeanSquare(StatisticalFunction):
    def compute(self, x:np.array):
        agg = self.get_numpy_aggregator()
        y = agg.global_avg(x)
        z=y-x
        return agg.global_avg(z**2)