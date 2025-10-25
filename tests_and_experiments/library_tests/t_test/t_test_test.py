
import numpy as np

from library.under_development.mann_whittney.mann_whittney_utest import MannWhitneyUTest
from tests_and_experiments.core.partitioned_table import PartitionedPandasTable
from tests_and_experiments.core.test_template import FederationTestTemplate
from library.under_development.t_test.t_test import OneSampleTtest,IndependentTtest,PairedTtest
import pandas as pd
import random
from scipy.stats import mannwhitneyu
from scipy import stats

def paired_ttest_manual(before, after):
    """
    Perform paired t-test manually
    """
    differences = np.array(after) - np.array(before)
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # t-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(n))

    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    return t_stat, p_value
class TTestTester(FederationTestTemplate):
    def __init__(self, client_id, client_count, *, dataset):
        super().__init__(client_id, client_count, dataset=dataset)

    def federated_computation(self, local_dataset):
        x = local_dataset['x'].values
        y = local_dataset['y'].values
        t_scipy1, p_scipy1 = OneSampleTtest(self.client).compute(x, 100)
        t_scipy, p_scipy = IndependentTtest(self.client).compute(x, y)
        t_paired, p_paired = PairedTtest(self.client).compute(x, y)
        return t_scipy1, p_scipy1, t_scipy, p_scipy, t_paired, p_paired

    def centralized_computation(self, centralized_dataset):
        x = centralized_dataset['x'].values
        y = centralized_dataset['y'].values
        t_scipy1, p_scipy1 = stats.ttest_1samp(x, 100)
        t_scipy, p_scipy = stats.ttest_ind(x, y, equal_var=True)
        t_paired, p_paired = stats.ttest_rel(x, y)
        return t_scipy1, p_scipy1,t_scipy, p_scipy, t_paired, p_paired

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)