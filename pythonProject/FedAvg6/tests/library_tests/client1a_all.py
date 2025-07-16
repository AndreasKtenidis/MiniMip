from tests.help_datasets.diabetes import DiabetesDiseaseDataset
from tests.help_datasets.job_training import JobTrainingDataset
from tests.help_datasets.metric import MetricDataset
from tests.help_datasets.titanic import TitanicDataset
from tests.help_datasets.titanic_as_disease import TitanicAsDiseaseDataset
from tests.help_datasets.wine_quality import WineQualityDataset
from tests.library_tests.group_comparisons.chisquared_fisher_test import ChiSquaredAndFisherTest
from tests.library_tests.ipwt.ipwt import IPWTTest
from tests.library_tests.linear_regression.linear_regression_test import LinearRegressionTest
from tests.library_tests.logistic_regression.logistic_regression_tests import LogisticRegressionTest
from tests.library_tests.median.MedianTest import MedianTest
from tests.library_tests.metric_tests.metric_tests import MetricTest
from tests.library_tests.multivariable_regression.multivariable_regression_test import MultivariableRegressionTest
from tests.library_tests.ordinal_logistic_regression.ordinal_logistic_regression_test import \
    OrdinalLogisticRegressionTest
from tests.library_tests.pandas_covariance.pandas_covariance_test import PandasCovarianceTest
from tests.library_tests.propensity_score.propensity_score_test import PropensityScoreTest
from tests.library_tests.standarized_mean_differences.smd_test import SmdTest
from tests.help_datasets.iris import IrisDataset
from tests.help_datasets.insuranse import InsuranceDataset

if __name__ == "__main__":
    # LogisticRegressionTest(0, 2, dataset = IrisDataset(),features=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
    #    'petal width (cm)'], target='target', operation_id=3)

    # IPWTTest(0, 2,dataset= TitanicAsDiseaseDataset(),treatment='Treatment',
    #                                     confounders=['pclass', 'age', 'sibsp', 'parch', 'fare'], operation_id=1)


    PropensityScoreTest(0, 2,
                        dataset=JobTrainingDataset(),
                        operation_id=9,
                        confounders = ['age', 'educ', 'married', 'nodegree', 're74', 're75', 'black', 'hispan', 'white'],
                        treatment = 'treat')

    # ChiSquaredAndFisherTest(0, 2, dataset =TitanicDataset(),operation_id=123)
    # MedianTest(0, 2,JobTrainingDataset(), operation_id=4)
    # MetricTest(0, 2,MetricDataset(), operation_id=5)
    # MultivariableRegressionTest(0, 2, DiabetesDiseaseDataset(), operation_id=6)
    # OrdinalLogisticRegressionTest(0, 2, WineQualityDataset(), operation_id=7)
    # PandasCovarianceTest(0, 2,IrisDataset(), operation_id=8)

    # SmdTest(0, 2,IrisDataset(), operation_id=10)
    # LinearRegressionTest(0, 2,InsuranceDataset(), operation_id=2)