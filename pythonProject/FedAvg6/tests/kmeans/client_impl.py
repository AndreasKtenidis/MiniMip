import numpy as np
from pythonProject.FedAvg6.library.stats.k_means import KMeans
from pythonProject.FedAvg6.system.client.grpc_agg_client import GRPCClient
from pythonProject.FedAvg6.system.client.aggregation_client import NumpyAggClient

np.random.seed(42)

client_data = [
    np.vstack([
        np.random.randn(100, 2) * 2.0 + np.array([0, 0]),
        np.random.randn(100, 2) * 2.0 + np.array([8, 8])
    ]),
    np.vstack([
        np.random.randn(100, 2) * 2.0 + np.array([0, 8]),
        np.random.randn(100, 2) * 2.0 + np.array([8, 0])
    ])
]

config = {
    'num_clients': 2
}

def compute(client_num):
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    agg = NumpyAggClient(client)

    kmeans = KMeans(client)
    kmeans.compute(client_data[client_num], k=4)
