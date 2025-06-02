import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
class InsuranceDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        # Load dataset from URL
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        df = pd.read_csv(url)
        # One-hot encode categorical features
        df_encoded=self.preprocess_insurance_data(df)
        return df_encoded


    @staticmethod
    def preprocess_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the insurance DataFrame and returns a fully transformed DataFrame
        with all columns (including 'charges').
        """

        df = df.copy()

        numeric_features = ["age", "bmi", "children","charges"]
        categorical_features = ["sex", "smoker", "region"]

        # Find binary categorical features (with exactly 2 unique values excluding NaNs)
        binary_cat_features = [col for col in categorical_features if df[col].nunique(dropna=True) == 2]
        multi_cat_features = [col for col in categorical_features if df[col].nunique(dropna=True) > 2]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        binary_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder())
        ])

        multi_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("binary_cat", binary_cat_pipeline, binary_cat_features),
            ("multi_cat", multi_cat_pipeline, multi_cat_features)
        ])

        X_features = df[numeric_features + categorical_features]
        X_transformed = preprocessor.fit_transform(X_features)

        # Column names
        multi_cat_cols = []
        if multi_cat_features:
            multi_cat_cols = preprocessor.named_transformers_["multi_cat"]["onehot"].get_feature_names_out(
                multi_cat_features)

        all_transformed_cols = (
                numeric_features +
                binary_cat_features +
                list(multi_cat_cols)
        )

        transformed_df = pd.DataFrame(X_transformed, columns=all_transformed_cols, index=df.index)

        return transformed_df

# dataset = InsuranceDataset(0,1)
# age = dataset.get_attribute('age')
# print(np.median(age))
