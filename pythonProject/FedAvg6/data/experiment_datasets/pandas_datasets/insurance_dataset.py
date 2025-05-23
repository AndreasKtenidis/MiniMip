import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from abc import ABC,abstractmethod
from data.abstract_table import AbstractTable
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
from sklearn.datasets import load_diabetes


class InsuranceDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        # Load dataset from URL
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        df = pd.read_csv(url)
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
        return df_encoded

