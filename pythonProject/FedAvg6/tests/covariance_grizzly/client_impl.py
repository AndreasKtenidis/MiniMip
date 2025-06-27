import duckdb
import os

from pythonProject.FedAvg6.library.stats.bivariate_statistics import CovarianceGrizzly
from pythonProject.FedAvg6.system.client.grpc_agg_client import GRPCClient
from pythonProject.FedAvg6.system.client.aggregation_client import PandasAggClient


# Grizzly version: returns a Grizzly DataFrame for the client's partition
import pythonProject.grizzly as grizzly
from pythonProject.grizzly.sqlgenerator import SQLGenerator
from pythonProject.grizzly.relationaldbexecutor import RelationalExecutor

class TmpDataset:
    def __init__(self, partition_id, num_partitions):
        db_path = os.path.join(os.path.dirname(__file__), f'../../data/grizzly_pandas_test/covariance_client{partition_id+1}.duckdb')
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


def compute(client_num):
    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    dataset = TmpDataset(client_num,config['num_clients']).get_local_dataset()
    agg = PandasAggClient(client)

    cov = CovarianceGrizzly(client).compute(dataset, x='x', y='y')
    print(f"[Covariance] Computed covariance from client {client_num}:\n{cov}")


