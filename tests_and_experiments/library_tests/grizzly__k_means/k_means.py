from library.core.statistical_function import StatisticalFunction
from library.utils.numpy_aggregator import NumpyAggregator

from tests_and_experiments.core.grizzly_test_template import GrizzlyFederationTestTemplate
import numpy as np
import pandas as pd
from library.utils.aggregation_client import AggregationClientInterface
import matplotlib.pyplot as plt


class KMeansGrizzlyTest(GrizzlyFederationTestTemplate):
    def federated_computation(self, conn, local_dataset):
        df = conn.execute("SELECT count(x)  FROM BlobDataset").fetchdf()

        cols = 2
        k = 3
        centroids = KMeans(self.client)
        conn.create_function('update_closest', centroids.update_cycle, return_type='DOUBLE')
        centroids.compute(conn,2,3)
        lala = centroids.get_centroids()

        # Query all points
        df = conn.execute("SELECT x, y FROM BlobDataset").fetchdf()

        # Plot
        plt.scatter(df["x"], df["y"], c="blue", marker="o")
        plt.scatter(lala[0], lala[1], c="red", marker="^", s=100, label="points")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Points from DuckDB")
        plt.grid(True)
        plt.show()

    def centralized_computation(self, centralized_dataset):
        return None

    def compare(self, federated_output, global_output):
        print("federated_output:", federated_output)

class KMeans(StatisticalFunction):
    def __init__(self, client: AggregationClientInterface):
        super().__init__(client)
        self._centroids=None
        self._new_centroids = None
        self._count = None
        self.aggregator =NumpyAggregator(client)

    def compute(self,conn, cols,k,*,max_iters=20):
        np.random.seed(42)
        self.centroids = np.random.uniform(low=-1.0, high=1.0, size=(k, cols))

        self._centroids = pd.DataFrame([(-5, -5), (-2, 10), (2, 2)])
        self._new_centroids = pd.DataFrame(np.zeros((k, cols)))
        self._count = pd.DataFrame(np.zeros((k, 1)))
        for _ in range(max_iters):
            conn.execute("SELECT x, y, update_closest(x, y) AS sum_xy FROM BlobDataset").fetchall()
            self.update(k,cols)

    def update_cycle(self, *args):
        distances = np.sqrt(((self._centroids - args) ** 2).sum(axis=1))
        i = distances.idxmin()
        self._new_centroids.iloc[i] = self._new_centroids.iloc[i]+args
        self._count.iloc[i] = self._count.iloc[i]+1
        return i

    def update(self,k,cols):
        # Avoid division by zero by replacing 0 with 1 (or handle differently)
        # Divide each row by the corresponding count
        global_centroids =pd.DataFrame(self.aggregator.fed_sum(self._new_centroids.values), columns=self._new_centroids.columns)
        global_count = pd.DataFrame(self.aggregator.fed_sum(self._count.values),columns=self._count.columns)

        #
        self._centroids = global_centroids.div(global_count[0], axis=0)
        self._new_centroids = pd.DataFrame(np.zeros((k, cols)))
        self._count = pd.DataFrame(np.zeros((k, 1)))

    def get_centroids(self):
        return self._centroids

