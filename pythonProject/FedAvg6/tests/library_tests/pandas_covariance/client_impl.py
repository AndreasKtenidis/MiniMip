import pandas as pd
import os

from pythonProject.FedAvg6.library.stats.bivariate_statistics import CovariancePandas
from pythonProject.FedAvg6.library.stats.bivariate_statistics import PearsonCorrelationPandas
from pythonProject.FedAvg6.library.stats.bivariate_statistics import LeastSquaresRegressionPandas
from pythonProject.FedAvg6.system.client.grpc_agg_client import GRPCClient
from pythonProject.FedAvg6.system.client.aggregation_client import PandasAggClient

class TmpDataset:
    def __init__(self, partition_id, num_partitions):
        csv_path = os.path.join(os.path.dirname(__file__), '../../data/grizzly_pandas_test/covariance_100mb.csv')
        df = pd.read_csv(csv_path)[['x', 'y']]
        n = len(df)
        size = n // num_partitions
        start = partition_id * size
        end = (partition_id + 1) * size if partition_id < num_partitions - 1 else n
        self.dataset = df.iloc[start:end]

    def get_local_dataset(self):
        return self.dataset


config = {
        'num_clients':2
    }


import time
def compute(client_num):
    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    start_global_time = time.time()
    dataset = TmpDataset(client_num,config['num_clients']).get_local_dataset()
    agg = PandasAggClient(client)

    start_time = time.time()
    cov=CovariancePandas(client).compute(dataset,x='x',y= 'y')
    end_time = time.time()
    print(f"Time taken for covariance computation on client {client_num}: {end_time - start_time:.5f} seconds")
    print(f"Computed covariance from client {client_num}:\n{cov}")
    start_time = time.time()
    pearson = PearsonCorrelationPandas(client).compute(dataset, x='x', y='y')
    end_time = time.time()
    print(f"Time taken for Pearson correlation computation on client {client_num}: {end_time - start_time:.5f} seconds")
    print(f"Computed Pearson correlation from client {client_num}:\n{pearson}")
    start_time = time.time()
    slope, intercept = LeastSquaresRegressionPandas(client).compute(dataset, x='x', y='y')
    end_time = time.time()
    print(f"Time taken for least squares regression computation on client {client_num}: {end_time - start_time:.5f} seconds")
    print(f"Computed least squares regression from client {client_num}:\n{slope}, {intercept}")
    end_global_time = time.time()
    print(f"Time taken with data loading, for all three algorithms from client {client_num}: {end_global_time - start_global_time:.5f} seconds")

