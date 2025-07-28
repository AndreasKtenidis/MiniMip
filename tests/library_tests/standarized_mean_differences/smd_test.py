
from library.templates.partitioned_table import PartitionedPandasTable
from tests.help_datasets.iris import IrisDataset
from tests.test_template.test_template import FederationTestTemplate

import numpy as np

class SmdTest(FederationTestTemplate):
    def federated_computation(self, local_dataset):
        from library.stats.bivariate_statistics import StandardizedMeanDifferences
        smd = StandardizedMeanDifferences(self.client).compute(local_dataset['sepal length (cm)'].values, local_dataset['petal width (cm)'].values)
        return smd

    def centralized_computation(self, centralized_dataset):
        attr1 = centralized_dataset['sepal length (cm)']
        attr2 = centralized_dataset['petal width (cm)']
        # Compute means
        mean1 = attr1.mean()
        mean2 = attr2.mean()
        # Compute sample standard deviations
        std1 = attr1.std(ddof=1)
        std2 = attr2.std(ddof=1)
        # Compute pooled standard deviation
        pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
        # Compute SMD
        smd = (mean1 - mean2) / pooled_std
        return smd

    def compare(self, federated_output, global_output):
        print('a',federated_output)
        print('b',global_output)