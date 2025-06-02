from library.stats.statistical_function import StatisticalFunction
import numpy as np
import matplotlib.pyplot as plt


class StandardHistogram(StatisticalFunction):

    def compute(self, x, num_bins):
        agg=self.get_numpy_aggregator()
        min_val = agg.global_min(x)
        max_val = agg.global_max(x)
        counts, bin_edges = np.histogram(x, bins=num_bins,range=(min_val, max_val))
        # StandardHistogram.plot_histogram(counts, bin_edges)
        counts = agg.fed_sum(counts)
        # StandardHistogram.plot_histogram(counts, bin_edges)
        return counts, bin_edges

    @staticmethod
    def plot_histogram(counts, bin_edges):
        # Plot manually using bar
        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_widths / 2

        plt.bar(bin_centers, counts, width=bin_widths, color='skyblue', edgecolor='black', align='center')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Simple Histogram from Counts & Bins')
        plt.show()
