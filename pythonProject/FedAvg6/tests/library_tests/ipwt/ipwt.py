from library.causal.ipwt import IPWT
import numpy as np

from tests.help_datasets.titanic_as_disease import TitanicAsDisease
from library.templates.partitioned_table import PartitionedPandasTable
from tests.test_template.test_template import FederationTestTemplate
from sklearn.linear_model import LogisticRegression

class IPWTTest(FederationTestTemplate):


    def federated_computation(self, local_dataset):
        ipwt = IPWT(self.client)
        federated_output = ipwt.compute(local_dataset, treatment='Treatment',
                                        confounders=['pclass', 'age', 'sibsp', 'parch', 'fare'])
        # Creating a global output from local outputs
        return federated_output

    def centralized_computation(self, centralized_output):
        confounders = ['pclass', 'age', 'sibsp', 'parch', 'fare']
        # 2. Estimate propensity scores: P(Treatment | Confounders)
        x = centralized_output[confounders]
        y = centralized_output['Treatment']
        logistic = LogisticRegression(max_iter=200)
        logistic.fit(x, y)
        centralized_output['ps'] = logistic.predict_proba(x)[:, 1]

        # Compute IPTW weights
        centralized_output['weight'] = np.where(
            centralized_output['Treatment'] == 1,
            1 / centralized_output['ps'],
            1 / (1 - centralized_output['ps'])
        )
        return centralized_output


    def get_partitioned_pandas_table(self) -> PartitionedPandasTable:
        return TitanicAsDisease()

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)