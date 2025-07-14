from sklearn.neighbors import NearestNeighbors

from library.templates.partitioned_table import PartitionedPandasTable
from tests.help_datasets.job_training import JobTrainingDataset
from tests.test_template.test_template import FederationTestTemplate
import pandas as pd
from library.causal.propensity_score2 import PropensityScore
from sklearn.linear_model import LogisticRegression

class PropensityScoreTest(FederationTestTemplate):

    def federated_computation(self, local_dataset):
        features = ['age', 'educ', 'married', 'nodegree', 're74', 're75'] + \
                   [col for col in local_dataset.columns if col.startswith('race_')]
        treatment = 'treat'
        propensity_score = PropensityScore(self.client)
        return propensity_score.compute(local_dataset, features=features, treatment=treatment)

    def centralized_computation(self, centralized_dataset):
        # Parameters
        features = ['age', 'educ', 'married', 'nodegree', 're74', 're75'] + \
                   [col for col in centralized_dataset.columns if col.startswith('race_')]
        treatment = 'treat'

        # Model Training
        x = centralized_dataset[features]
        y = centralized_dataset[treatment]
        model = LogisticRegression(max_iter=1000).fit(x, y)
        centralized_dataset['propensity_score'] = model.predict_proba(x)[:, 1]

        # Split treatment/control
        treated = centralized_dataset[centralized_dataset[treatment] == 1]
        control = centralized_dataset[centralized_dataset[treatment] == 0]

        # Match using Nearest Neighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control[['propensity_score']])
        distances, indices = nn.kneighbors(treated[['propensity_score']])
        matched_control = control.iloc[indices.flatten()]
        matched_df = pd.concat([treated.reset_index(drop=True), matched_control.reset_index(drop=True)], axis=0)
        return matched_control, matched_df

    def get_partitioned_pandas_table(self) -> PartitionedPandasTable:
        return JobTrainingDataset()

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)

