import sys

from tests_and_experiments.datasets.blob import BlobDataset
from tests_and_experiments.library_tests.grizzly_pearson.pearson_test import PearsonTest

def main():
    aggregation_server= None
    if len(sys.argv) < 2:
        aggregation_server="localhost:50051"
    else:
        aggregation_server = sys.argv[1]
        print(aggregation_server)


    PearsonTest(0, 2,
                           dataset=BlobDataset(),
                           operation_id=0,
                           aggregation_server=aggregation_server)



if __name__ == "__main__":
    main()