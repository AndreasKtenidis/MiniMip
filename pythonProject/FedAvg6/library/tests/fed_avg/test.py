from library.stat_models.statistical_Model import StatisticalModel
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
import numpy as np


class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        array_10 = [
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        data = pd.DataFrame(array_10)
        return data


config = {
        'batch_size': 64,
        'learning_rate': 0.01,
        'num_epochs': 20,
        'eval_every': 2,  # Evaluate every 2 epochs
        'num_clients':2
    }

class Lala(StatisticalModel):

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


def compute(client_num):
    # Creating Client
    client = GRPCClient(client_num, config['num_clients'], 1)

    data = TmpDataset(client_num,config['num_clients']).get_local_dataset()

    lala = Lala(client)
    agg = lala.get_numpy_aggregator()

    coeff = data.to_numpy()
    print("Coefficients:",coeff)
    weight = np.random.randint(1, 100)  # Upper bound is exclusive (11 means up to 10)
    print("Weight:",client_num)
    coeff = agg.fed_weighted_avg(coeff,1-client_num)
    print("Weighted Coefficients:",coeff)


