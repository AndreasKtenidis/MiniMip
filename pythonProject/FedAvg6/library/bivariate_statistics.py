from data.fed_table import FedTable
from data.pandas_federation.fed_multiset import Multiset
from function.abstract_function import AggFunc
import math

class Covariance(AggFunc):
    def compute(self, x:FedTable, y:FedTable):
        avg_x = x.fed_avg()
        avg_y = y.fed_avg()
        return ((x - avg_x) * (y - avg_y)).fed_avg()

class PearsonCorrelation(AggFunc):
    def compute(self, x:FedTable, y:FedTable):
        import pandas as pd
        x = Multiset(pd.concat([x, y], axis=1),client=x.get_client())
        print(x.fed_sum())
        print(x.fed_count())
        print(x.fed_avg())

        print(x.fed_min())
        print(x.fed_max())


        cov = Covariance().compute(x, y)
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
