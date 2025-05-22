import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    LabelEncoder
)

class TabularDataPreprocessor:
    """
    A flexible preprocessor for tabular data that handles:
    - Numeric features (scaling/imputation)
    - Categorical features (encoding/imputation)
    - Optional target scaling
    """

    def __init__(self,
                 numeric_features=None,
                 categorical_features=None,
                 numeric_strategy='scale',
                 categorical_strategy='onehot',
                 target_scaling=True):
        """
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            numeric_strategy: 'scale' (StandardScaler), 'minmax' (MinMaxScaler), or 'passthrough'
            categorical_strategy: 'onehot', 'ordinal', or 'passthrough'
            target_scaling: Whether to scale target variable
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.target_scaling = target_scaling
        self.feature_preprocessor = None
        self.target_scaler = None

    def _get_numeric_transformer(self):
        """Create pipeline for numeric features"""
        if self.numeric_strategy == 'scale':
            return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
        elif self.numeric_strategy == 'minmax':
            return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])
        return 'passthrough'

    def _get_categorical_transformer(self):
        """Create pipeline for categorical features"""
        if self.categorical_strategy == 'onehot':
            return Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ])
        elif self.categorical_strategy == 'ordinal':
            return Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
        return 'passthrough'

    def fit(self, x, y=None):
        """Fit preprocessing pipelines"""
        # Feature preprocessing
        self.feature_preprocessor = ColumnTransformer(
            transformers=[
                ('num', self._get_numeric_transformer(), self.numeric_features),
                ('cat', self._get_categorical_transformer(), self.categorical_features)
            ],
            remainder='drop'  # Drop columns not explicitly handled
        )
        self.feature_preprocessor.fit(x)

        # Target scaling
        if y is not None and self.target_scaling:
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(y.reshape(-1, 1))

        return self

    def transform(self, x, y=None):
        """Apply fitted transformations"""
        x_transformed = self.feature_preprocessor.transform(x)

        results = {'features': x_transformed}

        if y is not None and self.target_scaling:
            y_transformed = self.target_scaler.transform(y.reshape(-1, 1))
            results['target'] = y_transformed
        elif y is not None:
            results['target'] = y

        return results

    def inverse_transform_target(self, y):
        """Reverse target scaling if applicable"""
        if self.target_scaler:
            return self.target_scaler.inverse_transform(y)
        return y


# Example Usage
def execution():
    # Sample data
    data = {
        'age': [25, 30, None, 40],
        'income': [50000, None, 70000, 80000],
        'gender': ['M', 'F', 'M', None],
        'purchased': [1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    # Initialize preprocessor
    preprocessor = TabularDataPreprocessor(
        numeric_features=['age', 'income'],
        categorical_features=['gender'],
        numeric_strategy='scale',
        categorical_strategy='onehot',
        target_scaling=True
    )

    # Fit and transform
    x = df.drop('purchased', axis=1)
    y = df['purchased'].values
    preprocessor.fit(x, y)
    processed = preprocessor.transform(x, y)

    print("Processed Features:")
    print(processed['features'])
    print("\nProcessed Target:")
    print(processed['target'])

execution()