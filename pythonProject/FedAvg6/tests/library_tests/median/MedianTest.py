from library.stats.median import MedianBasedOnHistogram
from library.templates.partitioned_table import PartitionedPandasTable

from tests.help_datasets.job_training import JobTrainingDataset
from tests.test_template.test_template import FederationTestTemplate



class MediaTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        median_calc = MedianBasedOnHistogram(self.client)
        median = median_calc.compute(local_dataset['age'], num_bins=10)
        return median


    def centralized_computation(self, centralized_dataset):
        return centralized_dataset['age'].median()


    def get_partitioned_pandas_table(self) -> PartitionedPandasTable:
        return JobTrainingDataset()

    def compare(self, federated_output, global_output):
        print('federated_output', federated_output)
        print('global_output', global_output)


