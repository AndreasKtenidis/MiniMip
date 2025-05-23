from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing._encoders import _BaseEncoder, OneHotEncoder

from client.aggregation_client import AggregationClient
from function.abstract_function import AggFunc


class FedOneHotEncoder(AggFunc):
    def __init__(self,client:AggregationClient):
        super().__init__(client)
        self.encoder = OneHotEncoder()

    def compute(self,x):
        self.fit(x)
        print(x)


    def fit(self, x, y=None, **fit_params):
        aggregator = self.get_numpy_aggregator()
        if y is not None:
            y_agg = aggregator.fed_union(y)
        else:
            y_agg = None
        x_agg = aggregator.fed_union(x)
        self.encoder.fit(x_agg,y_agg,**fit_params)




