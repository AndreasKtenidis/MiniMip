from library.working_in_minimip.causal.ipwt import IPWT
import numpy as np

from tests.test_template.test_template import FederationTestTemplate
from sklearn.linear_model import LogisticRegression

class IPWTTest(FederationTestTemplate):
    def __init__(self, client_id, client_count, *, dataset, operation_id=0, treatment, confounders):
        self.treatment = treatment
        self.confounders = confounders
        super().__init__(client_id, client_count, dataset=dataset, operation_id=operation_id)

    def federated_computation(self, local_dataset):
        ipwt = IPWT(self.client)
        federated_output = ipwt.compute(local_dataset, treatment=self.treatment, confounders=self.confounders)
        # Creating a global output from local outputs
        return federated_output

    def centralized_computation(self, centralized_output):
        # 2. Estimate propensity scores: P(Treatment | Confounders)
        x = centralized_output[self.confounders]
        y = centralized_output[self.treatment]
        logistic = LogisticRegression(max_iter=200)
        logistic.fit(x, y)
        centralized_output['ps'] = logistic.predict_proba(x)[:, 1]

        # Compute IPTW weights
        centralized_output['weight'] = np.where(
            centralized_output[self.treatment] == 1,
            1 / centralized_output['ps'],
            1 / (1 - centralized_output['ps'])
        )
        return centralized_output

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)