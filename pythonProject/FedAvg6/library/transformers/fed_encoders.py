# from sklearn.preprocessing._encoders import OneHotEncoder
#
# from system.client.aggregation_client import AggregationClient
# from library.stats._statistical_function import StatisticalFunction
#
#
# class FedOneHotEncoder(StatisticalFunction):
#     def __init__(self,client:AggregationClient):
#         super().__init__(client)
#         self.encoder = OneHotEncoder()
#
#     def compute(self,x):
#         self.fit(x)
#         print(x)
#
#
#     def fit(self, x, y=None, **fit_params):
#         aggregator = self.get_numpy_aggregator()
#         if y is not None:
#             y_agg = aggregator.fed_union(y)
#         else:
#             y_agg = None
#         x_agg = aggregator.fed_union(x)
#         self.encoder.fit(x_agg,y_agg,**fit_params)




