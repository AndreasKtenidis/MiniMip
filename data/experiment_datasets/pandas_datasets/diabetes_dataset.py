# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# columns = [
#     "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
#     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
# ]
#
# df = pd.read_csv(url, names=columns)

import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
class DiabetesDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        # Load dataset from URL
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]

        df = pd.read_csv(url, names=columns)
        # One-hot encode categorical features
        df_encoded=self.preprocess_dataset(df)
        return df_encoded

    @staticmethod
    def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        # Separate the target
        target_column= 'Outcome'
        y = dataset[target_column]

        # Identify numeric and categorical features (excluding the target)
        numeric_features = dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if target_column in numeric_features:
            numeric_features.remove(target_column)

        categorical_features = dataset.select_dtypes(include=["object", "category"]).columns.tolist()

        # Specify ordinal features manually if needed
        ordinal_features = []  # e.g., ['EducationLevel']
        nominal_features = [col for col in categorical_features if col not in ordinal_features]

        # Define preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("ord", OrdinalEncoder(), ordinal_features),
                ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features)
            ],
            remainder="drop"
        )

        # Apply transformations
        X_processed = preprocessor.fit_transform(dataset.drop(columns=[target_column]))

        # Get column names for output DataFrame
        nom_cols = []
        if nominal_features:
            nom_cols = preprocessor.named_transformers_["nom"].get_feature_names_out(nominal_features).tolist()

        all_columns = numeric_features + ordinal_features + nom_cols

        # Construct the final DataFrame
        X_df = pd.DataFrame(X_processed, columns=all_columns)
        X_df[target_column] = y.reset_index(drop=True)

        return X_df


# titanic = DiabetesDataset(0,1)
# print(titanic.get_attribute_names())
# print(titanic.get_attribute_names())