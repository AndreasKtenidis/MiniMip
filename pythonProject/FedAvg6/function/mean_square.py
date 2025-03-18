from client.aggregation_client import NumpyAggregationClient
from function.abstract_function import AggFunc


class MeanSquare(AggFunc):
    def __init__(self,agg_client: NumpyAggregationClient ):
        super().__init__(agg_client)

    def compute(self, x):
        y = self.agg_client.avg(x)
        z=y-x
        return self.agg_client.avg(z**2)