from library.working_in_minimip.stat_models import FederatedLinearRegression
from tests.test_template.test_template import FederationTestTemplate
from sklearn.linear_model import LinearRegression

class LinearRegressionTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        x = local_dataset.drop('charges', axis=1).values
        y = local_dataset['charges'].values
        model = FederatedLinearRegression(self.client)
        model.fit(x, y)
        output = model.predict(x)
        return output

    def centralized_computation(self, centralized_dataset):
        x = centralized_dataset.drop('charges', axis=1).values # Features
        y = centralized_dataset['charges'].values
        model = LinearRegression()
        model.fit(x, y)
        output = model.predict(x)
        return output

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)

