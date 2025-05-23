from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
        """
        Preprocesses the Titanic DataFrame and returns a fully transformed DataFrame
        with all columns (including Survived if present).
        """

        df = df.copy()

        # Drop irrelevant columns
        df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], errors="ignore")

        # Define features including the target (we don't separate it)
        numeric_features = ["Age", "Fare"]
        categorical_features = ["Pclass", "Sex", "Embarked", "SibSp", "Parch"]

        # Pipelines
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        # Column transformer
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        # Fit and transform
        X_features = df[numeric_features + categorical_features]
        X_transformed = preprocessor.fit_transform(X_features)

        # Get transformed column names
        cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features)
        all_transformed_cols = np.concatenate([numeric_features, cat_cols])

        # Create new DataFrame from transformed features
        transformed_df = pd.DataFrame(X_transformed, columns=all_transformed_cols, index=df.index)

        # Add any remaining columns (like 'Survived') back
        remaining_cols = df.drop(columns=numeric_features + categorical_features)
        final_df = pd.concat([transformed_df, remaining_cols], axis=1)

        return final_df


# titanic = TitanicPandasDataset(1,2)
# print(titanic.get_attribute_names())
# print(titanic.get_attributes('PassengerId','Sex'))
