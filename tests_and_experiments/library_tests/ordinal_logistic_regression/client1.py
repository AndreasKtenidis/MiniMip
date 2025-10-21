
from tests_and_experiments.datasets.wine_quality import WineQualityDataset
from tests_and_experiments.library_tests.ordinal_logistic_regression.ordinal_logistic_regression_test import \
    OrdinalLogisticRegressionTest

def main():
    OrdinalLogisticRegressionTest(0, 2, dataset = WineQualityDataset(), operation_id=7)

if __name__ == "__main__":
    main()