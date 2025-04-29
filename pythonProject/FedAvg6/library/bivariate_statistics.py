from function.abstract_function import AggFunc
import math
import numpy as np

class Covariance(AggFunc):

    def compute(self, x:np.array, y:np.array):
        aggregator = self.get_numpy_aggregator()
        avg_x = aggregator.global_avg(x)
        avg_y = aggregator.global_avg(y)
        return aggregator.global_avg(((x - avg_x) * (y - avg_y)))

class PearsonCorrelation(AggFunc):
    def compute(self, x:np.array, y:np.array):
        aggregator = self.get_numpy_aggregator()
        cov = Covariance(self.client).compute(x, y)
        avg_data1=aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        stddev1 = math.sqrt(aggregator.global_avg(((x - avg_data1) ** 2)))
        stddev2 = math.sqrt(aggregator.global_avg(((y - avg_data2) ** 2)))
        return cov / (stddev1 * stddev2) if stddev1 > 0 and stddev2 > 0 else 0

class LeastSquaresRegression(AggFunc):
    def compute(self, x:np.array, y:np.array):
        cov = Covariance(self.client).compute(x, y)
        aggregator = self.get_numpy_aggregator()
        avg_data1 = aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        var_data1 = aggregator.global_avg(((x - avg_data1) ** 2))
        slope = cov / var_data1
        intercept = avg_data2 - slope * avg_data1
        return slope, intercept

class SumOfProducts(AggFunc):
    def compute(self, x:np.array, y:np.array):
        aggregator = self.get_numpy_aggregator()
        avg_data1 = aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        return aggregator.global_sum(((x - avg_data1) * (y - avg_data2)))
