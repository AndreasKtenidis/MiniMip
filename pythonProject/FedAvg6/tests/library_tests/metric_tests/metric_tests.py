from metrics.federated_metrics import LogisticRegressionFedMetrics
from tests.help_datasets.metric import MetricDataset
from tests.test_template.test_template import FederationTestTemplate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricTest(FederationTestTemplate):

    def compare(self, federated_output, global_output):
        print('local output:', federated_output)
        print('global output:', global_output)

    def federated_computation(self, local_dataset):
        y_true = local_dataset['y_true'].values
        y_prob = local_dataset['y_prob'].values
        y_pred = local_dataset['y_pred'].values
        # Compute metrics
        metrics = LogisticRegressionFedMetrics(self.client)
        accuracy = metrics.accuracy_score(y_true=y_true, y_pred= y_pred)
        precision = metrics.precision_score(y_true=y_true, y_pred= y_pred)
        recall = metrics.recall_score(y_true=y_true, y_pred= y_pred)
        f1 = metrics.f1_score(y_true=y_true,y_pred= y_pred)
        auc_score = metrics.auc_score(y_true=y_true, y_prob=y_prob)
        return accuracy,precision,recall,f1,auc_score


    def centralized_computation(self, centralized_dataset):
        y_true = centralized_dataset['y_true'].values
        y_prob = centralized_dataset['y_prob'].values
        y_pred = centralized_dataset['y_pred'].values
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_score =roc_auc_score(y_true, y_prob)
        return accuracy,precision,recall,f1,auc_score
