

import numpy as np
from scipy import stats

from library.core.statistical_function import StatisticalFunction
from library.under_development.stats.univariate_statistics import StandardDeviation,Variance
from library.utils.numpy_aggregator import NumpyAggregator


class OneSampleTtest(StatisticalFunction):
    def compute(self, x: np.array,pop_mean:float):
        agg = NumpyAggregator(self.client)
        n = agg.global_count(x)
        sample_mean = agg.global_avg(x)
        sample_std = StandardDeviation(self.client).compute(x, ddof=1)
        # Calculate t-statistic
        t_stat = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        return t_stat, p_value

class IndependentTtest(StatisticalFunction):
    def compute(self, group1: np.array,group2: np.array):
        agg = NumpyAggregator(self.client)
        """
            Perform independent two-sample t-test manually
            """
        n1 = agg.global_count(group1)
        n2 = agg.global_count(group2)
        mean1 = agg.global_avg(group1)
        mean2 = agg.global_avg(group2)
        var1 = Variance(self.client).compute(group1, ddof=1)
        var2 = Variance(self.client).compute(group2, ddof=1)
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        # Standard error
        std_error = pooled_std * np.sqrt(1 / n1 + 1 / n2)
        # t-statistic
        t_stat = (mean1 - mean2) / std_error
        # Degrees of freedom
        df = n1 + n2 - 2
        # p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        return t_stat, p_value

class PairedTtest(StatisticalFunction):
    def compute(self, a: np.array, b: np.array):
        """
        Perform paired t-test manually
        """
        agg = NumpyAggregator(self.client)
        differences = a-b
        n = agg.global_count(differences)
        mean_diff = agg.global_avg(differences)
        std_diff = StandardDeviation(self.client).compute(differences, ddof=1)
        # t-statistic
        t_stat = mean_diff / (std_diff / np.sqrt(n))
        # p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        return t_stat, p_value

