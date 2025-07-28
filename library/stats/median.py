from library.templates.statistical_function import StatisticalFunction
import numpy as np
from pandas.core.series import Series

from library.stats.histogram import StandardHistogram


class MedianBasedOnHistogram(StatisticalFunction):

    def compute(self, x: Series,*, num_bins=10):

        hist = StandardHistogram(self.client)
        counts, bin_edges = hist.compute(x.values, num_bins)
        return MedianBasedOnHistogram._compute_median_from_histogram(counts, bin_edges)

    @staticmethod
    def _compute_median_from_histogram(counts, bin_edges):
        """
        Compute the median from a histogram given bin_edges and counts.

        :param bin_edges: List of bin edges. Length = num_bins + 1.
        :param counts: List of bin counts. Length = num_bins.
        :return: Median value.
        """
        total_count = sum(counts)
        half_total = total_count / 2

        cumulative = 0
        for i, count in enumerate(counts):
            cumulative += count
            if cumulative >= half_total:
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                bin_width = bin_end - bin_start

                cumulative_before_bin = cumulative - count
                median_offset = (half_total - cumulative_before_bin) / count

                median = bin_start + median_offset * bin_width
                return median
        return None
