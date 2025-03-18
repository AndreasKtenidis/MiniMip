from client.aggregation_client import NumpyAggregationClient
from function.abstract_function import AggFunc


class MeanSquare(AggFunc):

    def compute(self, x):
        y = self.agg_client.avg(x)
        z=y-x
        return self.agg_client.avg(z**2)

#
#
# def variance(self):
#     """Returns the variance of the dataset."""
#     if self.n > 1:
#         sum_of_squares = sum(x ** 2 for x in self.data)
#         return (sum_of_squares / self.n) - (self.avg_data ** 2)
#     return 0  # Variance is 0 if there's only 1 data point
#
# def standard_deviation(self):
#     """Returns the standard deviation of the dataset."""
#     return math.sqrt(self.variance())
#
# def sum_of_squares(self):
#     """Returns the sum of squares of the dataset."""
#     return sum(x ** 2 for x in self.data)
#
# def range(self):
#     """Returns the range of the dataset (max - min)."""
#     return max(self.data) - min(self.data)
#
# def coefficient_of_variation(self):
#     """Returns the coefficient of variation (std_dev / mean)."""
#     if self.avg_data != 0:
#         return self.standard_deviation() / self.avg_data
#     return 0  # If the mean is 0, the coefficient of variation is undefined
#
# def mean_absolute_deviation(self):
#     """Returns the mean absolute deviation (MAD) of the dataset."""
#     return sum(abs(x - self.avg_data) for x in self.data) / self.n if self.n > 0 else 0
#
# def root_mean_square(self):
#     """Returns the root mean square (RMS) of the dataset."""
#     return math.sqrt(sum(x ** 2 for x in self.data) / self.n) if self.n > 0 else 0
