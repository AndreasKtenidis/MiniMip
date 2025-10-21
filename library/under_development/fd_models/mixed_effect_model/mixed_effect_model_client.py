from mini_mip_system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
from library.under_development.fd_models.mixed_effect_model.mixed_effect_model import RandomEffects


class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/plm/Grunfeld.csv"
        data = pd.read_csv(url)
        return data


config = {
    'batch_size': 64,
    'learning_rate': 0.01,
    'num_epochs': 20,
    'eval_every': 2,  # Evaluate every 2 epochs
    'num_clients': 2
}


def compute(client_num):
    # Creating Client
    client = GRPCClient(client_num, config['num_clients'], client_num)
    #  Creating the Data
    data = TmpDataset(client_num, config['num_clients']).get_local_dataset()
    #
    # Clean and prepare data
    data = data.rename(columns={'inv': 'invest'})  # Rename 'inv' to 'invest' for clarity
    data = data.set_index(['firm', 'year'])  # Assuming 'firm' and 'year' are in the data

    # Run model with correct column names
    model = RandomEffects(
        data['invest'],  # Dependent variable (formerly 'inv')
        data[['value', 'capital']],
        # Predictors
        weights=None
    ).fit(cov_type='robust')

    print(model)

    print(model.params)  # Returns a pandas Series
