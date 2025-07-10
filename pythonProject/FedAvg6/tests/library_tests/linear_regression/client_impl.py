import numpy as np

from library.stat_models.linear_regression_ols import FedOLS


from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        insurance_df = pd.read_csv(url)
        return insurance_df


config = {
        'num_clients':2
    }

def compute(client_num):
    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)



    insurance_df = TmpDataset(client_num,config['num_clients']).get_local_dataset()
    insurance_processed = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)
    # Define features (X) and target (y)
    X = insurance_processed.drop('charges', axis=1)  # Features
    y = insurance_processed['charges']

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = FedOLS(client)
    model.fit(X_train.values, y_train.values)

    # Predict on test set
    y_pred = model.predict(X_test.values)


    # Evaluate model
    # TODO nead to create some federated metrics\
    from system.client.aggregation_client import NumpyAggClient
    agg = NumpyAggClient(client)
    error = np.array(mean_squared_error(y_test.values, y_pred))
    mse = agg.fed_weighted_avg(error,len(y_test.values))
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
