import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset

class TitanicPandasDataset(FederatedPandasDataset):



    def get_dataset(self):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        dataset = pd.read_csv(url)
        return dataset



