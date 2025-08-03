import sys

from tests.help_datasets.blob import BlobDataset
from tests.library_tests.k_means_rand_Pandas.kmeans_test_pandas import KmeansPandasTest


def main():
    aggregation_server= None
    if len(sys.argv) < 2:
        aggregation_server="localhost:50051"
    else:
        aggregation_server = sys.argv[1]
        print(aggregation_server)


    KmeansPandasTest(0, 2,
                           dataset=BlobDataset(),
                           features=['x','y'],
                           operation_id=0,
                           aggregation_server=aggregation_server)



if __name__ == "__main__":
    main()