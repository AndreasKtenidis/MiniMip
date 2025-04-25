from data.fed_table import FedTable
from function.abstract_function import AggFunc
import math

class Covariance(AggFunc):
    def compute(self, x:FedTable, y:FedTable):
        avg_x = x.fed_avg()
        avg_y = y.fed_avg()
        return ((x - avg_x) * (y - avg_y)).fed_avg()

class PearsonCorrelation(AggFunc):
    def compute(self, x:FedTable, y:FedTable):
        cov = Covariance().compute(x, y)
        print('cov',cov)
        avg_data1=x.fed_avg()
        avg_data2 = y.fed_avg()
        stddev1 = math.sqrt(((x - avg_data1) ** 2).fed_avg())
        stddev2 = math.sqrt(((y - avg_data2) ** 2).fed_avg())
        return cov / (stddev1 * stddev2) if stddev1 > 0 and stddev2 > 0 else 0

class LeastSquaresRegression(AggFunc):
    def compute(self, x:FedTable, y:FedTable):
        cov = Covariance().compute(x, y)
        avg_data1 = (x.fed_avg())
        avg_data2 = (y.fed_avg())
        var_data1 = ((x - avg_data1) ** 2).fed_avg()
        slope = cov / var_data1
        intercept = avg_data2 - slope * avg_data1
        return slope, intercept

class SumOfProducts(AggFunc):
    def compute(self, x:FedTable, y:FedTable):
        avg_data1 = x.fed_avg()
        avg_data2 = y.fed_avg()
        return ((x - avg_data1) * (y - avg_data2)).fed_sum()
