from pythonProject.FedAvg6.data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TitanicPandasDataset(FederatedPandasDataset):



    def get_dataset(self):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        dataset = pd.read_csv(url)
        return dataset



