import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import duckdb
from tests.help_datasets.blob import BlobDataset
from tests.test_template.grizzly_table_factory import GrizzlyFactory
from duckdb.typing import INTEGER

# # Step 1: Load synthetic blob dataset
# points, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0, random_state=38)
# points = pd.DataFrame(points, columns=['x', 'y'])
#
# k = 3

class KMeansPandas():

    def compute(self, dataset, columns, k):
        collection_of_centroids = CentroidCollection( columns, k)
        duckdb.create_function(
            name='find_nearest',
            function=collection_of_centroids.find_nearest,
            return_type=INTEGER
        )

        # Step 3: Pure Pandas k-means loop
        for iteration in range(1):
            print(collection_of_centroids.centroids)
            dataset['cluster'] = collection_of_centroids.find_nearest(dataset,columns)
            print(dataset.generateQuery())
            print(dataset.show())



            centroid_sum_max = dataset.groupby('cluster').sum('x')
            print(centroid_sum_max.generateQuery())
            print(centroid_sum_max.show())

        return collection_of_centroids





class CentroidCollection:
    def __init__(self,  columns, k:int):
        self.centroids: pd.DataFrame = pd.DataFrame(np.random.uniform(low=-1.0, high=1.0, size=(k, len(columns))), columns=columns)
        self.k = k

    def find_nearest(self, dataset,columns):
        expression = []

        for i in range(len(columns)):
            tmp =  " + ".join(f"(dataset['{n}']-{self.centroids.iloc[i][n]})" for n in columns)
            expression.append(f"{(tmp)}")
        expr = eval(f"np.argmax(np.array({expression}))")
        # expr = f"np.argmax(np.array([0,0,7]))"
        print(expr.generateQuery())
        return eval(expr)





        # distances = np.sqrt((self.centroids['x'][0] - x)**2 + (self. centroids['y'][1] - y)**2)
        # return distances

    # def find_closest_centroid(self, point_row):
    #     centroids = self.centroids
    #     point = point_row.values[0:centroids.shape[1]].reshape(1, -1)
    #     distances = cdist(point, centroids.values, metric='euclidean')
    #     closest_idx = np.argmin(distances)
    #     return centroids.index[closest_idx]


def compute_new_centroids(summary: pd.DataFrame):
    cols = list(summary.columns)
    n_rows = len(summary)
    n_features = len(cols) // 2  # assuming pairs of columns (_sum and _count)

    # Initialize output array: shape (n_rows, n_features)
    out = np.zeros((n_rows, n_features))

    for i in range(0, len(cols) - 1, 2):
        feature_idx = i // 2
        sum_col = cols[i]
        count_col = cols[i + 1]
        out[:, feature_idx] = summary[sum_col].values / summary[count_col].values
    return pd.DataFrame(out)

fact=GrizzlyFactory()
dataset = fact.create_db(BlobDataset(),client_id=0,num_clients=1)
# dataset = BlobDataset().get_global_dataset()
KMeansPandas().compute(dataset,['x','y'],3)

# pandas = KMeansPandas()
# pandas.compute(dataset, 3)

# import duckdb
# from duckdb.typing import INTEGER
#
# class MyFuncs:
#     def __init__(self,id):
#         self.id = id
#
#     def square(self, x: int) -> int:
#         return x * x*self.id
#
# funcs = MyFuncs(3)
#
# duckdb.create_function(
#     name='square',
#     function=funcs.square,
#     parameters=[INTEGER],            # Optional but helpful
#     return_type=INTEGER              # ✅ Use 'return_type', not 'returns'
# )
#
# print(duckdb.query("SELECT square(8)").fetchall())  # [(64,)]
#
# KMeansPandas().compute(dataset,3)