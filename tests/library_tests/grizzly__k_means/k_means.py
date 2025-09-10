from library.templates.statistical_function import StatisticalFunction
from tests.help_datasets.blob import BlobDataset
from tests.test_template.grizzly_test_template import GrizzlyFederationTestTemplate
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def map_to_three(x:float,y:float) -> float:
    return x+y

class PearsonTest2(GrizzlyFederationTestTemplate):
    def federated_computation(self,conn, local_dataset):

        pearson_value = NoServer(self.client).compute(local_dataset)
        return pearson_value

    def centralized_computation(self, centralized_dataset):
        return None

    def compare(self, federated_output, global_output):
        print("federated_output:", federated_output)

class NoServer(StatisticalFunction):
    def compute(self, points):
        print("!!!!!!!!")
        points['lala'] = points[['x','y']].map(map_to_three)
        print(points.collect())


        # Create the mapped column and apply alias
        #

        # return points
        # mapped_result.evaluate()
        # print(type(mapped_result))
        # constant_column = mapped_result.alias('constant_value')
        #
        # # Add to DataFrame
        # result_df = points.with_columns(constant_column)
        # computed_df = result_df.compute()

    def initialize_centroids(self, X: np.ndarray, k: int) -> np.ndarray:
            """Generate random centroids and federate a global initialization."""
            n_features = X.shape[0]
            np.random.seed(42)
            return  np.random.uniform(low=-1.0, high=1.0, size=(k, n_features))

class CentroidCollection:

    def __init__(self, X: pd.DataFrame, k: int) -> None:
        """Generate random centroids and federate a global initialization."""
        n_features = X.shape[0]
        np.random.seed(42)
        self.centroids= np.random.uniform(low=-1.0, high=1.0, size=(k, n_features))

    #
    # def find_closest_centroid(point_row, centroids):
    #     point = point_row.values[0:centroids.shape[1]].reshape(1, -1)
    #     distances = cdist(point, centroids.values, metric='euclidean')
    #     closest_idx = np.argmin(distances)
    #     return centroids.index[closest_idx]

aggregation_server="localhost:50051"

PearsonTest2(0, 2,
                           dataset=BlobDataset(),
                           operation_id=0,
                           aggregation_server=aggregation_server)