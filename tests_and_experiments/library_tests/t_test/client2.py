from tests_and_experiments.datasets.MannWhittneyDataset import MannWhittneyDataset
from tests_and_experiments.library_tests.mann_whittney_u.mann_whittney_tester import MannWhittneyTester
from tests_and_experiments.library_tests.t_test.t_test_test import TTestTester

TTestTester(1, 2, dataset=MannWhittneyDataset())

