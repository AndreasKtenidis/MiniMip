from library.under_development.stats.bivariate_statistics import CovariancePandas
from tests_and_experiments.core.test_template import FederationTestTemplate


class PandasCovarianceTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        cov = CovariancePandas(self.client)
        out = cov.compute(local_dataset, x='sepal length (cm)', y='sepal width (cm)')
        return out

    def centralized_computation(self, centralized_dataset):
        return centralized_dataset['sepal length (cm)'].cov(centralized_dataset['sepal width (cm)'])

    def compare(self, federated_output, global_output):
        print('fed',federated_output)
        print('global',global_output)