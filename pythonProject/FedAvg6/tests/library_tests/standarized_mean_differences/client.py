
from sklearn.datasets import load_iris
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset


class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        return df


config = {
        'num_clients':2
    }

def compute(client_num):
    # # Creating Client
    client = GRPCClient(client_num, config['num_clients'], client_num)
    df = TmpDataset(client_num,config['num_clients']).get_local_dataset()

    from library.stats.bivariate_statistics import StandardizedMeanDifferences
    smd = StandardizedMeanDifferences(client).compute(df['sepal length (cm)'].values,df['petal width (cm)'].values)
    print(smd)