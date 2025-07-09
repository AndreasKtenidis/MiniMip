import numpy as np

from library.stat_models.linear_regression_ols import FedOLS


from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from tests.multivariable_regression.test import multivariable_regression
from tests.testing_datasets.diabetes import DiabetesDisease

config = {
        'num_clients':2
    }

def compute(client_num):
    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    #
    diabetes = DiabetesDisease()
    local_data = diabetes.get_local_dataset(client_num,config['num_clients'])
    local_x = local_data[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']].values
    local_y = local_data[['target']].values
    #
    global_data = diabetes.get_dataset()
    global_x = global_data[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']].values
    global_y = global_data[['target']].values

    # Creating Federated Model
    model1 = FedOLS(client)
    model1.fit(local_x, local_y)
    # Creating Non Federated Model
    from sklearn.linear_model import LinearRegression
    model2 =  LinearRegression()
    model2.fit(global_x, global_y)
    #
    print(model1.predict(global_x))
    print(model2.predict(global_x))
