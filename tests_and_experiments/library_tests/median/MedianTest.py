from library.under_development.stats.median import MedianBasedOnHistogram
from tests_and_experiments.core.test_template import FederationTestTemplate

class MedianTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        median_calc = MedianBasedOnHistogram(self.client)
        median = median_calc.compute(local_dataset['age'], num_bins=10)
        return median

    def centralized_computation(self, centralized_dataset):
        return centralized_dataset['age'].median()

    def compare(self, federated_output, global_output):
        print('federated_output', federated_output)
        print('global_output', global_output)