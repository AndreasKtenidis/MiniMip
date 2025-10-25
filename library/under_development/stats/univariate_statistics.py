
import numpy as np

from library.core.statistical_function import StatisticalFunction
from library.utils.numpy_aggregator import NumpyAggregator


class Dummy(StatisticalFunction):
    def compute(self, x):
        random_integers = np.random.randint(1, 11, size=3)
        print(random_integers)
        return 3


class Variance(StatisticalFunction):
    def compute(self, x: np.array,*,ddof=0):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        agg = NumpyAggregator(self.client)
        n = agg.global_count(x)
        if n == 0:
            raise ValueError("Data array cannot be empty")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        if n <= ddof:
            raise ValueError(f"Not enough data points for ddof={ddof}. Need at least {ddof + 1} points.")

        mean = agg.global_avg(x)
        sum_squared_diff = agg.global_sum((x - mean) ** 2)
        variance = sum_squared_diff / (n - ddof)
        return variance


class StandardDeviation(StatisticalFunction):
    def compute(self, x: np.array,*,ddof=0):
        variance = Variance(self.client).compute(x, ddof=ddof)
        return np.sqrt(variance)


class SumOfSquares(StatisticalFunction):
    def compute(self, x: np.array):
        agg = NumpyAggregator(self.client)
        return agg.global_sum(x ** 2)


class Range(StatisticalFunction):
    def compute(self, x: np.array):
        agg = NumpyAggregator(self.client)
        return agg.global_max(x) - agg.global_min(x)


class CoefficientOfVariation(StatisticalFunction):
    def compute(self, x: np.array):
        agg = NumpyAggregator(self.client)
        avg_data = agg.global_avg(x)
        if avg_data == 0:
            return 0
        StandardDeviation(self.client).compute(x) / avg_data


class MeanAbsoluteDeviation(StatisticalFunction):
    def compute(self, x: np.array):
        agg = NumpyAggregator(self.client)
        if agg.global_count(x) == 0:
            return 0
        avg_data = agg.global_avg(x)
        return agg.global_avg(abs(x - avg_data))


class RootMeanSquare(StatisticalFunction):
    def compute(self, x: np.array):
        agg = NumpyAggregator(self.client)
        if agg.global_count(x) == 0:
            return 0
        return agg.global_avg(np.sqrt(x ** 2))


class MeanSquare(StatisticalFunction):
    def compute(self, x: np.array):
        agg = NumpyAggregator(self.client)
        y = agg.global_avg(x)
        z = y - x
        return agg.global_avg(z ** 2)
