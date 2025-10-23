
import numpy as np

from library.under_development.mann_whittney.mann_whittney_utest import MannWhitneyUTest
from tests_and_experiments.core.partitioned_table import PartitionedPandasTable
from tests_and_experiments.core.test_template import FederationTestTemplate

import pandas as pd
import random
from scipy.stats import mannwhitneyu

class MannWhittneyTester(FederationTestTemplate):
    def __init__(self, client_id, client_count, *, dataset):
        super().__init__(client_id, client_count, dataset=dataset)

    def federated_computation(self, local_dataset):
        x = local_dataset[['x']].values
        y = local_dataset[['y']].values
        func = MannWhitneyUTest(self.client)
        return func.compute(x, y,num_bins=100)

    def centralized_computation(self, centralized_dataset):
        # 2. Estimate propensity scores: P(Treatment | Confounders)
        x = centralized_dataset[['x']].values
        y = centralized_dataset[['y']].values
        print(len(x))
        return mannwhitneyu(x, y, use_continuity=True, alternative='two-sided', method='asymptotic')

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)