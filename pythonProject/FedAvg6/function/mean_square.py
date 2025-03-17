from function.abstract_function import AggFunc


class MeanSquare(AggFunc):
    def compute(self, x):
        y = self.agg_client.avg(x)
        z=y-x
        return self.agg_client.avg(z**2)