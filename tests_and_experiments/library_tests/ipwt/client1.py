import sys

from tests_and_experiments.datasets.titanic_as_disease import TitanicAsDiseaseDataset
from tests_and_experiments.library_tests.ipwt.ipwt_test import IPWTTest


def main():
    aggregation_server= None
    if len(sys.argv) < 2:
        aggregation_server="localhost:50051"
    else:
        aggregation_server = sys.argv[1]
        print(aggregation_server)

    IPWTTest(0, 2, dataset=TitanicAsDiseaseDataset(), treatment='Treatment',
             confounders=['pclass', 'age', 'sibsp', 'parch', 'fare'], operation_id=1)



if __name__ == "__main__":
    main()