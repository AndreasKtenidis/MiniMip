from tests.test_template.test_template import FederationTestTemplate

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from library.working_in_minimip.stat_models import FedOLS


class MultivariableRegressionTest(FederationTestTemplate):
    def federated_computation(self, local_dataset):
        # Creating input and output
        x = local_dataset[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']].values
        y = local_dataset[['target']].values
        # Train the model
        model = FedOLS(self.client)
        model.fit(x, y)
        return model.predict(x)

    def centralized_computation(self, centralized_dataset):
        x = centralized_dataset[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']].values
        y = centralized_dataset[['target']].values
        # Add intercept term
        x = sm.add_constant(x)
        # Fit OLS regression
        model = LinearRegression()
        model.fit(x, y)
        return model.predict(x)

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)