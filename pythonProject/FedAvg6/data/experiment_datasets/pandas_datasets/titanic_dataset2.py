from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
import pandas as pd

class TitanicPandasDataset2(FederatedPandasDataset):



    def get_dataset(self):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        return pd.read_csv(url)


titanic = TitanicPandasDataset2(1,2)
print(titanic.get_attribute_names())
print(titanic.get_attributes('PassengerId','Sex'))
