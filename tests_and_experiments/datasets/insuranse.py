from library.core.partitioned_table import PartitionedPandasTable
import pandas as pd
from library.core.partitioned_table import PartitionedPandasTable
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class InsuranceDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        # Load the insurance dataset (replace with your file path)
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        insurance_df = pd.read_csv(url)
        # Convert categorical variables (sex, smoker, region) into numerical using one-hot encoding
        insurance_processed = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)
        return insurance_processed

