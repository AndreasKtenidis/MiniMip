from library.causal.ipwt import IPWT
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from tests.ipwt.test import ipwt_non_federated
from library.templates.partitioned_table import PartitionedPandasTable


class TmpDataset(PartitionedPandasTable):


    def get_dataset(self) -> pd.DataFrame:
        return sns.load_dataset('titanic')

def compute(client_num):

    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    # Federated Dataset
    data =dataset.get_local_dataset(client_num,config['num_clients'])
    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    ipwt = IPWT(client)
    federated_output = ipwt.compute(data,treatment='sex',outcome='survived',confounders = ['pclass', 'age', 'sibsp', 'parch', 'fare'] )
    # Creating a global output from local outputs
    non_federated_output = ipwt_non_federated(sns.load_dataset('titanic'))
    print(federated_output)
    print(non_federated_output)

dataset = TmpDataset()
config = {'num_clients':2}


