from typing import List

from abstract_function import RoundCall, AggFunction


class Round1(RoundCall):
    def compute(self, x) -> List[AggFunction]:
        y = x ** 2
        pass

    # def compute(self, x):
    #     y = x ** 2
    #     return self.agg_client.avg(y)