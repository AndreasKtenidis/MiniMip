from library.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.templates.partitioned_table import PartitionedPandasTable

from metrics.federated_metrics import FederatedMetrics
from tests.test_template.test_template import FederationTestTemplate
from tests.help_datasets.iris import IrisDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import concurrent.futures

class LogisticRegressionTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        iris =local_dataset
        x = iris.drop('target', axis=1).values
        y = iris['target'].values

        # Train logistic regression model
        model = FederatedLogisticRegressionClientSaSo(self.client)
        model.fit(x, y)

        # Predict and evaluate
        y_pred = model.predict(x)
        y_prob = model.predict_proba(x)

        # Compute metrics
        metrics = FederatedMetrics(self.client)
        accuracy = metrics.accuracy(y, y_pred)
        precision = metrics.precision(y, y_pred)
        recall = metrics.recall(y, y_pred)
        f1 = metrics.f1_score(y, y_pred)
        auc_score = metrics.auc_score(y_true=y, y_prob=y_prob[:, 1])
        return accuracy, precision, recall, f1, auc_score

    def centralized_computation(self, centralized_dataset):
        iris = centralized_dataset
        x = iris.drop('target', axis=1).values
        y = iris['target'].values

        # Train logistic regression model
        model = LogisticRegression()
        model.fit(x, y)

        # Predict and evaluate
        y_pred = model.predict(x)
        y_prob = model.predict_proba(x)

        # Compute Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc_score = roc_auc_score(y, y_prob[:, 1])
        return accuracy, precision, recall, f1, auc_score

    def get_partitioned_pandas_table(self) -> PartitionedPandasTable:
        return IrisDataset()

    def compare(self, local_output, global_output):
        print(local_output)
        print(global_output)


if __name__ == "__main__":
    client_count = 2  # Define your client count

    with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
        # Create instances of MyClass in each thread
        futures = [executor.submit(LogisticRegressionTest, client_id, client_count)
                   for client_id in range(client_count)]
        concurrent.futures.wait(futures)