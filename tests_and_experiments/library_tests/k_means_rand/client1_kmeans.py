from tests_and_experiments.library_tests.k_means_rand.kmeans_test import KMeansTest
from tests_and_experiments.datasets.blob import BlobDataset


def main():
    KMeansTest(0, 2,
                           dataset=BlobDataset(),
                           operation_id=0,
                           aggregation_server=None)



if __name__ == "__main__":
    main()