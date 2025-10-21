import sys

from tests_and_experiments.library_tests.grizzly__k_means.k_means import KMeansGrizzlyTest
from tests_and_experiments.datasets.blob import BlobDataset


def main():
    aggregation_server= None
    if len(sys.argv) < 2:
        aggregation_server="localhost:50051"
    else:
        aggregation_server = sys.argv[1]
        print(aggregation_server)





if __name__ == "__main__":
    main()