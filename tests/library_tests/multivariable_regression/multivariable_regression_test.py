from library.causal.ipwt import IPWT
import numpy as np

from tests.help_datasets.titanic_as_disease import TitanicAsDiseaseDataset
from library.templates.partitioned_table import PartitionedPandasTable
from tests.test_template.test_template import FederationTestTemplate
from sklearn.linear_model import LogisticRegression


import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

from library.stat_models.linear_regression_ols import FedOLS


from mini_mip_system.client.grpc_agg_client import GRPCClient

from tests.help_datasets.diabetes import DiabetesDiseaseDataset

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