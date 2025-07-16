from mord import LogisticIT
from sklearn.neighbors import NearestNeighbors

from library.stat_models.ordinal_logistic_regression import FedOrdinalLogisticRegression
from library.stats.bivariate_statistics import CovariancePandas
from library.templates.partitioned_table import PartitionedPandasTable
from system.client.aggregation_client import PandasAggClient
from tests.help_datasets.iris import IrisDataset
from tests.help_datasets.job_training import JobTrainingDataset
from tests.help_datasets.metric import MetricDataset
from tests.help_datasets.wine_quality import WineQualityDataset
from tests.test_template.test_template import FederationTestTemplate


class PandasCovarianceTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        cov = CovariancePandas(self.client)
        out = cov.compute(local_dataset, x='sepal length (cm)', y='sepal width (cm)')
        return out

    def centralized_computation(self, centralized_dataset):
        return centralized_dataset['sepal length (cm)'].cov(centralized_dataset['sepal width (cm)'])

    def compare(self, federated_output, global_output):
        print('fed',federated_output)
        print('global',global_output)