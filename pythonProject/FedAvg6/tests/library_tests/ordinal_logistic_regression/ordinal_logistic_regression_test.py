from mord import LogisticIT
from sklearn.neighbors import NearestNeighbors

from library.stat_models.ordinal_logistic_regression import FedOrdinalLogisticRegression
from library.templates.partitioned_table import PartitionedPandasTable
from tests.help_datasets.job_training import JobTrainingDataset
from tests.help_datasets.wine_quality import WineQuality
from tests.test_template.test_template import FederationTestTemplate
import pandas as pd
from library.causal.propensity_score import PropensityScore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class OrdinalLogisticRegressionTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        X = local_dataset.drop(columns=['quality']).values
        y = local_dataset['quality'].values  # quality is ordinal: 3-8 (integers)

        # Because the dataset is imbalanced and for simplicity,
        # we will only use quality scores 3 to 7 and discard 8.
        mask = y <= 7
        X = X[mask]
        y = y[mask]
        # Creating model
        model = FedOrdinalLogisticRegression(client=self.client)
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)
        # Evaluation
        print("Classification Report:\n")
        print(classification_report(y, y_pred, zero_division=0))

    def centralized_computation(self, centralized_dataset):
        # Features and target
        X = centralized_dataset.drop(columns=['quality']).values
        y = centralized_dataset['quality'].values  # quality is ordinal: 3-8 (integers)
        # Because the dataset is imbalanced and for simplicity,
        # we will only use quality scores 3 to 7 and discard 8.
        mask = y <= 7
        X = X[mask]
        y = y[mask]
        # Fit ordinal logistic regression model
        model = LogisticIT()
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)
        # Evaluation
        print("Classification Report:\n")
        print(classification_report(y, y_pred, zero_division=0))

    def get_partitioned_pandas_table(self) -> PartitionedPandasTable:
        return WineQuality()

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)