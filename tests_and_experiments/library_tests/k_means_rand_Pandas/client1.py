from tests_and_experiments.datasets.blob import BlobDataset
from tests_and_experiments.library_tests.k_means_rand_Pandas.kmeans_test_pandas import KmeansPandasTest


KmeansPandasTest(0, 2,
                           dataset=BlobDataset(),
                           features=['x','y'],
                           operation_id=0)