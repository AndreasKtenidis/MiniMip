import duckdb
import os

from pythonProject.FedAvg6.library.stats.bivariate_statistics import CovarianceGrizzly
from pythonProject.FedAvg6.library.stats.bivariate_statistics import PearsonCorrelationGrizzly
from pythonProject.FedAvg6.library.stats.bivariate_statistics import LeastSquaresRegressionGrizzly
from pythonProject.FedAvg6.system.client.grpc_agg_client import GRPCClient
from pythonProject.FedAvg6.system.client.aggregation_client import PandasAggClient


# Grizzly version: returns a Grizzly DataFrame for the client's partition
import pythonProject.grizzly as grizzly
from pythonProject.FedAvg6.tests.covariance_grizzly.db_setup import base_dir
from pythonProject.grizzly.sqlgenerator import SQLGenerator
from pythonProject.grizzly.relationaldbexecutor import RelationalExecutor

class TmpDataset:
    def __init__(self, partition_id, num_partitions):
        db_path = (base_dir /f"covariance_client{partition_id+1}.duckdb").resolve()
        print("db_path", db_path)

        self.con = duckdb.connect(database=db_path, read_only=False)
        gen = SQLGenerator("duckdb")
        executor = RelationalExecutor(self.con, gen)
        grizzly.use(executor)
        self.df = grizzly.read_table("covariance")

        # Filter to take only the columns we need
        self.df = self.df[['x', 'y']]

    def get_local_dataset(self):
        return self.df


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
    cov = CovarianceGrizzly(client).compute(dataset, x='x', y='y')
    end_time = time.time()
    print(f"Time taken for covariance computation on client {client_num}: {end_time - start_time:.5f} seconds")
    print(f"Computed covariance from client {client_num}:\n{cov}")
    start_time = time.time()
    pearson = PearsonCorrelationGrizzly(client).compute(dataset, x='x', y='y')
    end_time = time.time()
    print(f"Time taken for Pearson correlation computation on client {client_num}: {end_time - start_time:.5f} seconds")
    print(f"Computed Pearson correlation from client {client_num}:\n{pearson}")
    start_time = time.time()
    slope, intercept = LeastSquaresRegressionGrizzly(client).compute(dataset, x='x', y='y')
    end_time = time.time()
    print(f"Time taken for least squares regression computation on client {client_num}: {end_time - start_time:.5f} seconds")
    print(f"Computed least squares regression from client {client_num}:\n{slope}, {intercept}")
    end_global_time = time.time()
    print(f"Time taken with data loading, for all three algorithms from client {client_num}: {end_global_time - start_global_time:.5f} seconds")