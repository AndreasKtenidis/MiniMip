from library.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.stats.bivariate_statistics import CovariancePandas
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from system.client.aggregation_client import PandasAggClient

class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        iris = load_iris()
        rng = np.random.RandomState(42)  # For reproducibility
        shuffled_indices = rng.permutation(len(iris.data))

        iris.data = iris.data[shuffled_indices]
        iris.target = iris.target[shuffled_indices]


        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Filter to only use two classes (0 and 1)
        df = df[df['target'] != 2]
        return df


config = {
        'num_clients':2
    }


def compute(client_num):
    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    dataset = TmpDataset(client_num,config['num_clients']).get_local_dataset()
    agg = PandasAggClient(client)

    cov=CovariancePandas(client).compute(dataset,x='sepal length (cm)',y= 'sepal width (cm)')
    print(cov)


