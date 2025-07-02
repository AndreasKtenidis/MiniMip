import numpy as np

from library.causal.ipwt import IPWT
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
from sklearn.preprocessing import LabelEncoder

import seaborn as sns

class TmpDataset(FederatedPandasDataset):
    def get_dataset(self) -> pd.DataFrame:
        return sns.load_dataset('titanic')


config = {
        'num_clients':2
    }

def compute(client_num):
    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    # Federated Dataset
    data = TmpDataset(client_num,config['num_clients']).get_local_dataset()
    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    ipwt = IPWT(client)
    output = ipwt.compute(data,treatment='sex',outcome='survived',confounders = ['pclass', 'age', 'sibsp', 'parch', 'fare'] )

    print(output)