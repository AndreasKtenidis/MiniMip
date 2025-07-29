from library.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.templates.partitioned_table import PartitionedPandasTable

from metrics.federated_metrics import LogisticRegressionFedMetrics
from tests.test_template.test_template import FederationTestTemplate
from tests.help_datasets.iris import IrisDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



class LogisticRegressionTest(FederationTestTemplate):

    def __init__(self, client_id, client_count, *, dataset, operation_id=0,aggregation_server="localhost:50051", features, target):
        self.features = features
        self.target = target
        super().__init__(client_id, client_count, dataset=dataset, operation_id=operation_id,aggregation_server=aggregation_server)

    def federated_computation(self, local_dataset):
        x = local_dataset[self.features].values
        y = local_dataset[self.target].values

        # Train logistic regression model
        model = FederatedLogisticRegressionClientSaSo(self.client)
        model.fit(x, y)

        # Predict and evaluate
        y_pred = model.predict(x)
        y_prob = model.predict_proba(x)

        # Compute metrics
        metrics = LogisticRegressionFedMetrics(self.client)
        accuracy = metrics.accuracy_score(y_true=y, y_pred= y_pred)
        precision = metrics.precision_score(y_true=y, y_pred= y_pred)
        recall = metrics.recall_score(y_true=y, y_pred= y_pred)
        f1 = metrics.f1_score(y_true=y,y_pred= y_pred)
        auc_score = metrics.auc_score(y_true=y, y_prob=y_prob[:, 1])
        return accuracy, precision, recall, f1, auc_score

    def centralized_computation(self, centralized_dataset):
        x = centralized_dataset[self.features].values
        y = centralized_dataset[self.target].values

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

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)

