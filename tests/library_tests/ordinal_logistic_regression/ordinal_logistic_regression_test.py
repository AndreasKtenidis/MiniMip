from mord import LogisticIT

from library.working_in_minimip.stat_models import FedOrdinalLogisticRegression
from tests.test_template.test_template import FederationTestTemplate
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

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)