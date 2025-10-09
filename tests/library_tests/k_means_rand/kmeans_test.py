from library.working_in_minimip.stats import KMeansRand
from tests.test_template.test_template import FederationTestTemplate


class KMeansTest(FederationTestTemplate):

    def __init__(self, client_id, client_count, *, dataset, operation_id=0,  aggregation_server):
        super().__init__(client_id, client_count, dataset=dataset, operation_id=operation_id,aggregation_server=aggregation_server)


    def federated_computation(self, local_dataset):

        k_means = KMeansRand(self.client)
        return k_means.compute(local_dataset.values, 3)

    def centralized_computation(self, centralized_dataset):
        return None
        # # Parameters
        # # confounders = ['age', 'educ', 'married', 'nodegree', 're74', 're75'] + \
        # #            [col for col in centralized_dataset.columns if col.startswith('race_')]
        # # treatment = 'treat'
        #
        # # Model Training
        # x = centralized_dataset[self.confounders]
        # y = centralized_dataset[self.treatment]
        # model = LogisticRegression(max_iter=1000).fit(x, y)
        # centralized_dataset['propensity_score'] = model.predict_proba(x)[:, 1]
        #
        # # Split treatment/control
        # treated = centralized_dataset[centralized_dataset[self.treatment] == 1]
        # control = centralized_dataset[centralized_dataset[self.treatment] == 0]
        #
        # # Match using Nearest Neighbors
        # nn = NearestNeighbors(n_neighbors=1)
        # nn.fit(control[['propensity_score']])
        # distances, indices = nn.kneighbors(treated[['propensity_score']])
        # matched_control = control.iloc[indices.flatten()]
        # matched_df = pd.concat([treated.reset_index(drop=True), matched_control.reset_index(drop=True)], axis=0)
        # return matched_control, matched_df

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)

