from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
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
        prep_dataset = self.preprocess_titanic_data(dataset)
        return prep_dataset


    @staticmethod
    def preprocess_titanic_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop irrelevant columns
        df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], errors="ignore")

        # Define numeric and categorical features
        numeric_features = ["Age", "Fare"]

        # Determine binary vs non-binary categorical columns
        potential_categorical = ["Pclass", "Sex", "Embarked", "SibSp", "Parch"]
        binary_categorical = [col for col in potential_categorical if df[col].nunique() == 2]
        multiclass_categorical = [col for col in potential_categorical if df[col].nunique() > 2]

        # Pipelines
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        binary_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder())
        ])

        multiclass_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        # Column transformer
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("bin_cat", binary_cat_pipeline, binary_categorical),
            ("multi_cat", multiclass_cat_pipeline, multiclass_categorical)
        ])

        # Fit and transform
        X_features = df[numeric_features + binary_categorical + multiclass_categorical]
        X_transformed = preprocessor.fit_transform(X_features)

        # Get column names
        bin_cat_cols = binary_categorical
        multi_cat_cols = preprocessor.named_transformers_["multi_cat"]["encoder"].get_feature_names_out(
            multiclass_categorical)
        all_transformed_cols = np.concatenate([numeric_features, bin_cat_cols, multi_cat_cols])

        # Create transformed DataFrame
        transformed_df = pd.DataFrame(X_transformed, columns=all_transformed_cols, index=df.index)

        # Add remaining columns (e.g., "Survived")
        remaining_cols = df.drop(columns=numeric_features + binary_categorical + multiclass_categorical)
        final_df = pd.concat([transformed_df, remaining_cols], axis=1)

        return final_df


titanic = TitanicPandasDataset(1,2)
print(titanic.get_attribute_names())
print(titanic.get_attribute_names())
# print(titanic.get_attributes('PassengerId','Sex'))
