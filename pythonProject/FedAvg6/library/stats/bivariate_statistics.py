from library.stats.statistical_function import StatisticalFunction
import math
import numpy as np
from library.stats.univariate_statistics import StandardDeviation
class Covariance(StatisticalFunction):

    def compute(self, x:np.array, y:np.array):
        aggregator = self.get_numpy_aggregator()
        avg_x = aggregator.global_avg(x)
        avg_y = aggregator.global_avg(y)
        return aggregator.global_avg(((x - avg_x) * (y - avg_y)))

class PearsonCorrelation(StatisticalFunction):
    def compute(self, x:np.array, y:np.array):
        aggregator = self.get_numpy_aggregator()
        cov = Covariance(self.client).compute(x, y)
        avg_data1=aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        stddev1 = math.sqrt(aggregator.global_avg(((x - avg_data1) ** 2)))
        stddev2 = math.sqrt(aggregator.global_avg(((y - avg_data2) ** 2)))
        return cov / (stddev1 * stddev2) if stddev1 > 0 and stddev2 > 0 else 0

class LeastSquaresRegression(StatisticalFunction):
    def compute(self, x:np.array, y:np.array):
        cov = Covariance(self.client).compute(x, y)
        aggregator = self.get_numpy_aggregator()
        avg_data1 = aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        var_data1 = aggregator.global_avg(((x - avg_data1) ** 2))
        slope = cov / var_data1
        intercept = avg_data2 - slope * avg_data1
        return slope, intercept

class SumOfProducts(StatisticalFunction):
    def compute(self, x:np.array, y:np.array):
        aggregator = self.get_numpy_aggregator()
        avg_data1 = aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        return aggregator.global_sum(((x - avg_data1) * (y - avg_data2)))

class StandardizedMeanDifferences(StatisticalFunction):
    def compute(self, x:np.array, y:np.array):
        aggregator = self.get_numpy_aggregator()
        # Calculate means
        mean1 = aggregator.global_avg(x)
        mean2 = aggregator.global_avg(y)
        # Calculate standard deviations
        sd1 = StandardDeviation(self.client).compute(x)
        sd2 = StandardDeviation(self.client).compute(y)
        # Calculate sample sizes
        n1 = aggregator.global_count(x)
        n2 = aggregator.global_count(y)
        # Calculate pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
        # Calculate SMD
        smd = (mean1 - mean2) / pooled_sd
        return smd