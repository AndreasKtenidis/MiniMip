import numpy as np
import pandas as pd
from pythonProject.FedAvg6.library.stats.k_means import KMeansPandas
from pythonProject.FedAvg6.system.client.grpc_agg_client import GRPCClient
from pythonProject.FedAvg6.system.client.aggregation_client import PandasAggClient

np.random.seed(42)

# client_data = [
#     pd.DataFrame(
#         np.vstack([
#             np.random.randn(100, 2) * 2.0 + np.array([0, 0]),
#             np.random.randn(100, 2) * 2.0 + np.array([8, 8])
#         ]),
#         columns=['feature_1', 'feature_2']
#     ),
#     pd.DataFrame(
#         np.vstack([
#             np.random.randn(100, 2) * 2.0 + np.array([0, 8]),
#             np.random.randn(100, 2) * 2.0 + np.array([8, 0])
#         ]),
#         columns=['feature_1', 'feature_2']
#     )
# ]

client_data = [
    pd.DataFrame(
        [[0, 0], [10, 10]],  # Two points, clearly in different clusters
        columns=['feature_1', 'feature_2']
    ),
    pd.DataFrame(
        [[0, 10], [10, 0]],  # Two more points, also clearly separated
        columns=['feature_1', 'feature_2']
    )
]

config = {
    'num_clients': 2
}

def compute(client_num):
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    agg = PandasAggClient(client)

    print("Client data:\n", client_data)

    kmeans = KMeansPandas(client)
    print(f"Client {client_num} starting K-Means computation with {len(client_data[client_num])} data points.")
    kmeans.compute(client_data[client_num], k=2)
    print(f"Client {client_num} finished computation.")
