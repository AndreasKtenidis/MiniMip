from tests.library_tests.group_comparisons.chisquared_fisher_test import ChiSquaredAndFisherTest
from tests.library_tests.ipwt.ipwt import IPWTTest
from tests.library_tests.linear_regression.linear_regression_test import LinearRegressionTest
from tests.library_tests.logistic_regression.logistic_regression_tests import LogisticRegressionTest
from tests.library_tests.median.MedianTest import MediaTest
from tests.library_tests.metric_tests.metric_tests import MetricTest
from tests.library_tests.multivariable_regression.multivariable_regression_test import MultivariableRegressionTest
from tests.library_tests.ordinal_logistic_regression.ordinal_logistic_regression_test import \
    OrdinalLogisticRegressionTest
from tests.library_tests.pandas_covariance.pandas_covariance_test import PandasCovarianceTest
from tests.library_tests.propensity_score.propensity_score_test import PropensityScoreTest
from tests.library_tests.standarized_mean_differences.smd_test import SmdTest

ChiSquaredAndFisherTest(1,2)
# IPWTTest(1,2)
# LinearRegressionTest(1,2)
# LogisticRegressionTest(1,2)
# MediaTest(1,2)
# MetricTest(1,2)
# MultivariableRegressionTest(1,2)
# OrdinalLogisticRegressionTest(1,2)
# PandasCovarianceTest(1,2)
# PropensityScoreTest(1,2)
# SmdTest(1,2)
