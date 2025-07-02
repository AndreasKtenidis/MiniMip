from pythonProject.FedAvg6 import data
from pythonProject.FedAvg6.library.templates.statistical_function import StatisticalFunction
import math
import numpy as np
from library.stats.univariate_statistics import StandardDeviation


class Covariance(StatisticalFunction):

    def compute(self, x: np.array, y: np.array):
        aggregator = self.get_numpy_aggregator()
        avg_x = aggregator.global_avg(x)
        avg_y = aggregator.global_avg(y)
        return aggregator.global_avg(((x - avg_x) * (y - avg_y)))


class PearsonCorrelation(StatisticalFunction):
    def compute(self, x: np.array, y: np.array):
        aggregator = self.get_numpy_aggregator()
        cov = Covariance(self.client).compute(x, y)
        avg_data1 = aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        stddev1 = math.sqrt(aggregator.global_avg(((x - avg_data1) ** 2)))
        stddev2 = math.sqrt(aggregator.global_avg(((y - avg_data2) ** 2)))
        return cov / (stddev1 * stddev2) if stddev1 > 0 and stddev2 > 0 else 0


class LeastSquaresRegression(StatisticalFunction):
    def compute(self, x: np.array, y: np.array):
        cov = Covariance(self.client).compute(x, y)
        aggregator = self.get_numpy_aggregator()
        avg_data1 = aggregator.global_avg(x)
        avg_data2 = aggregator.global_avg(y)
        var_data1 = aggregator.global_avg(((x - avg_data1) ** 2))
        slope = cov / var_data1
        intercept = avg_data2 - slope * avg_data1
        return slope, intercept


# class SumOfProducts(StatisticalFunction):
#     def compute(self, x:np.array, y:np.array):
#         aggregator = self.get_numpy_aggregator()
#         avg_data1 = aggregator.global_avg(x)
#         avg_data2 = aggregator.global_avg(y)
#         return aggregator.global_sum(((x - avg_data1) * (y - avg_data2)))

from pandas import DataFrame


class CovariancePandas(StatisticalFunction):
    def compute(self, data: DataFrame, *, x, y):
        aggregator = self.get_pandas_aggregator()

        # Calculate global averages and subtract them from the data
        avg_data_x_y = aggregator.global_avg(data[[x,y]])
        tmp = data[[x, y]] - avg_data_x_y.values
        tmp2 = tmp.prod(axis=1).to_frame('__covariance'+x+y)

        # Get sum of cross-products and total count
        sum_of_products = aggregator.global_sum(tmp2)
        total_count = aggregator.global_count(data[[x]])

        covariance = sum_of_products.iloc[0, 0] / (total_count.iloc[0, 0] - 1)
        return covariance

class CovarianceGrizzly(StatisticalFunction):
    def compute(self, data:DataFrame,*, x,y):
        aggregator = self.get_grizzly_aggregator()

        # Calculate global averages and subtract them from the data
        avg_data_x_y = aggregator.global_avg(data[[x, y]])
        data['product'] = (data[x] - avg_data_x_y[0]) * (data[y] - avg_data_x_y[1])

        # Get sum of cross-products and total count
        sum_of_products = aggregator.global_sum(data[['product']])
        total_count = aggregator.global_count(data[[x]])
        
        covariance = sum_of_products[0] / (total_count - 1)
        return covariance

class PearsonCorrelationPandas(StatisticalFunction):
    def compute(self, data: DataFrame, *, x, y):
        aggregator = self.get_pandas_aggregator()
        cov = CovariancePandas(self.client).compute(data, x=x, y=y)
        avg_x, avg_y = aggregator.global_avg(data[[x, y]]).values[0]

        std_x = np.sqrt(aggregator.global_sum((data[[x]] - avg_x) ** 2).iloc[0, 0] / (aggregator.global_count(data[[x]]).iloc[0, 0] - 1))
        std_y = np.sqrt(aggregator.global_sum((data[[y]] - avg_y) ** 2).iloc[0, 0] / (aggregator.global_count(data[[y]]).iloc[0, 0] - 1))
        if std_x > 0 and std_y > 0:
            return cov / (std_x * std_y)
        else:
            return 0

class PearsonCorrelationGrizzly(StatisticalFunction):
    def compute(self, data, *, x, y):
        aggregator = self.get_grizzly_aggregator()
        cov = CovarianceGrizzly(self.client).compute(data, x=x, y=y)
        avg_data_x_y = aggregator.global_avg(data[[x, y]])

        data['square_x'] = (data[x] - avg_data_x_y[0]) * (data[x] - avg_data_x_y[0])
        data['square_y'] = (data[y] - avg_data_x_y[1]) * (data[y] - avg_data_x_y[1])

        sum_of_squares_x = aggregator.global_sum(data[['square_x']])
        sum_of_squares_y = aggregator.global_sum(data[['square_y']])
        total_count = aggregator.global_count(data[[x]])

        std_x = np.sqrt(sum_of_squares_x[0] / (total_count - 1))
        std_y = np.sqrt(sum_of_squares_y[0] / (total_count - 1))
        if std_x > 0 and std_y > 0:
            return cov / (std_x * std_y)
        else:
            return 0

class LeastSquaresRegressionPandas(StatisticalFunction):
    def compute(self, data: DataFrame, *, x, y):
        aggregator = self.get_pandas_aggregator()
        cov = CovariancePandas(self.client).compute(data, x=x, y=y)
        avg_x, avg_y = aggregator.global_avg(data[[x, y]]).values[0]

        sum_of_squares_x = aggregator.global_sum((data[[x]] - avg_x) ** 2).iloc[0, 0]
        total_count = aggregator.global_count(data[[x]]).iloc[0, 0]

        var_x = sum_of_squares_x / (total_count - 1)
        slope = cov / var_x
        intercept = avg_y - slope * avg_x
        return slope, intercept

class LeastSquaresRegressionGrizzly(StatisticalFunction):
    def compute(self, data: DataFrame, *, x, y):
        aggregator = self.get_grizzly_aggregator()
        cov = CovarianceGrizzly(self.client).compute(data, x=x, y=y)
        avg_data_x_y = aggregator.global_avg(data[[x, y]])

        data['square_x'] = (data[x] - avg_data_x_y[0]) * (data[x] - avg_data_x_y[0])
        sum_of_squares_x = aggregator.global_sum(data[['square_x']])
        total_count = aggregator.global_count(data[[x]])

        var_x = sum_of_squares_x[0] / (total_count - 1)
        slope = cov / var_x
        intercept = avg_data_x_y[1] - slope * avg_data_x_y[0]
        return slope, intercept


class StandardizedMeanDifferences(StatisticalFunction):
    def compute(self, x: np.array, y: np.array):
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
