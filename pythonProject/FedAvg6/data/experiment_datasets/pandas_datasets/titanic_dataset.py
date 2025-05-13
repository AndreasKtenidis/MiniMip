from data.experiment_datasets.pandas_datasets.federated_pandas_dataset import FederatedPandasDataset
import pandas as pd

class TitanicPandasDataset(FederatedPandasDataset):



    def get_dataset(self):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        return pd.read_csv(url)


titanic = TitanicPandasDataset(1,2)
print(titanic.get_attribute_names())
print(titanic.get_attribute('PassengerId'))
