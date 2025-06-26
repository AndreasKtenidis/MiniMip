from system.client.aggregation_client import PandasAggClient

from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset


class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        data = {
            'A': [10, 20, 30, 40],
            'B': [1.5, 2.5, 3.5, 4.5],
            'C': [100, 200, 300, 400]
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        return df


config = {
        'num_clients':2
    }

def compute(client_num):
    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    dataset = TmpDataset(client_num,config['num_clients']).get_local_dataset()
    agg = PandasAggClient(client)

    # print(agg.global_min(dataset))
    # print(agg.global_max(dataset))
    # print(agg.global_count(dataset))
    print(agg.global_avg(dataset))

