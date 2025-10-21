from tests_and_experiments.library_tests.standarized_mean_differences.smd_test import SmdTest
from tests_and_experiments.datasets.iris import IrisDataset

SmdTest(0, 2,dataset=IrisDataset(), operation_id=10)
