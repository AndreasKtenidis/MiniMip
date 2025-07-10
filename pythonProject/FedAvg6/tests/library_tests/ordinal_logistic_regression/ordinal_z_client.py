
from library.stat_models.ordinal_logistic_regression import FedOrdinalLogisticRegression
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
from sklearn.metrics import classification_report

class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        data = pd.read_csv(url, sep=';')
        return data


config = {
        'batch_size': 64,
        'learning_rate': 0.01,
        'num_epochs': 20,
        'eval_every': 2,  # Evaluate every 2 epochs
        'num_clients':2
    }

def compute(client_num):
    # Creating Client
    client = GRPCClient(client_num, config['num_clients'], client_num)




    data = TmpDataset(client_num,config['num_clients']).get_local_dataset()

    X = data.drop(columns=['quality']).values
    y = data['quality'].values  # quality is ordinal: 3-8 (integers)

    # Because the dataset is imbalanced and for simplicity,
    # we will only use quality scores 3 to 7 and discard 8.
    mask = y <= 7

    # X = NumpyWrapper(X[mask],client)
    # y = NumpyWrapper(y[mask],client)




    # Creating model
    model = FedOrdinalLogisticRegression(client=client)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)

    # Evaluation
    print("Classification Report:\n")
    print(classification_report(y, y_pred, zero_division=0))

