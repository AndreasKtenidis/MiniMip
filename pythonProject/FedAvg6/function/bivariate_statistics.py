from function.abstract_function import AggFunc
import math

class Covariance(AggFunc):
    def compute(self, x, y):
        avg_x = self.avg(x)
        avg_y = self.avg(y)
        return self.avg((x - avg_x) * (y - avg_y))

class PearsonCorrelation(AggFunc):
    def compute(self, x, y):
        cov = Covariance(self.agg_client).compute(x, y)
        avg_data1=self.avg(x)
        avg_data2 = self.avg(y)
        stddev1 = math.sqrt(self.avg((x - avg_data1) ** 2))
        stddev2 = math.sqrt(self.avg((y - avg_data2) ** 2))
        return cov / (stddev1 * stddev2) if stddev1 > 0 and stddev2 > 0 else 0

class LeastSquaresRegression(AggFunc):
    def compute(self, x, y):
        cov = Covariance(self.agg_client).compute(x, y)
        avg_data1 = self.avg(x)
        avg_data2 = self.avg(y)
        var_data1 = self.avg((x - avg_data1) ** 2)
        slope = cov / var_data1
        intercept = avg_data2 - slope * avg_data1
        return slope, intercept

class SumOfProducts(AggFunc):
    def compute(self, x, y):
        avg_data1 = self.avg(x)
        avg_data2 = self.avg(y)
        return self.sum((x - avg_data1) * (y - avg_data2))

# def spearman_rank_correlation(self):
#     """Returns the Spearman rank correlation coefficient between two variables."""
#     rank_data1 = [sorted(self.data1).index(x) + 1 for x in self.data1]
#     rank_data2 = [sorted(self.data2).index(x) + 1 for x in self.data2]
#
#     diff_squared = sum((r1 - r2) ** 2 for r1, r2 in zip(rank_data1, rank_data2))
#     return 1 - (6 * diff_squared) / (self.n * (self. n* *2 - 1))


