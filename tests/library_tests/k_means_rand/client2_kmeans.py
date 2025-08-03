import sys


from tests.library_tests.k_means_rand.kmeans_test import KMeansTest
from tests.help_datasets.blob import BlobDataset


def main():
    aggregation_server= None
    if len(sys.argv) < 2:
        aggregation_server="localhost:50051"
    else:
        aggregation_server = sys.argv[1]
        print(aggregation_server)

    KMeansTest(1, 2,
                           dataset=BlobDataset(),
                           operation_id=0,
                           aggregation_server=aggregation_server)



if __name__ == "__main__":
    main()