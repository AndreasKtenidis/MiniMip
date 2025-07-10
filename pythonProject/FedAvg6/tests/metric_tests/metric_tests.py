from metrics.federated_metrics import FederatedMetrics
from tests.help_datasets.auc_dataset import AucDataset
from tests.test_template.test_template import FederationTestTemplate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricTest(FederationTestTemplate):

    def compare(self, local_output, global_output):
        print('local output:', local_output)
        print('global output:', global_output)

    def local_computation(self):
        y_true = self.local_dataset['y_true'].values
        y_prob = self.local_dataset['y_prob'].values
        y_pred = self.local_dataset['y_pred'].values
        # Compute metrics
        metrics = FederatedMetrics(self.client)
        accuracy = metrics.accuracy(y_true, y_pred)
        precision = metrics.precision(y_true, y_pred)
        recall = metrics.recall(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        auc_score = metrics.auc_score(y_true=y_true, y_prob=y_prob)
        return accuracy,precision,recall,f1,auc_score


    def global_computation(self):
        y_true = self.global_dataset['y_true'].values
        y_prob = self.global_dataset['y_prob'].values
        y_pred = self.global_dataset['y_pred'].values
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_score =roc_auc_score(y_true, y_prob)
        return accuracy,precision,recall,f1,auc_score

    def get_partitioned_pandas_table(self):
        return AucDataset()