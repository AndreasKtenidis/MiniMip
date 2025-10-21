from tests_and_experiments.datasets.diabetes import DiabetesDiseaseDataset
from tests_and_experiments.library_tests.multivariable_regression.multivariable_regression_test import MultivariableRegressionTest



def main():
    MultivariableRegressionTest(0, 2, dataset = DiabetesDiseaseDataset(), operation_id=6)


if __name__ == "__main__":
    main()