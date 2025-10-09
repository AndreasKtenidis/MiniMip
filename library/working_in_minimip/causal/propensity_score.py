import pandas as pd
from sklearn.neighbors import NearestNeighbors

from library.working_in_minimip.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.templates.statistical_function import StatisticalFunction


class PropensityScore(StatisticalFunction):

    def compute(self,df,*,features, treatment):
        agg = self.get_numpy_aggregator()
        # Estimate propensity scores
        X = df[features]
        y = df[treatment]
        # Estimates a federated model
        model = FederatedLogisticRegressionClientSaSo(self.client)
        model.fit(X,y)
        df['propensity_score'] = model.predict_proba(X)[:, 1]
        # Split treatment/control
        treated = df[df[treatment] == 1]
        control = df[df[treatment] == 0]
        # Match using Nearest Neighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control[['propensity_score']])
        distances, indices = nn.kneighbors(treated[['propensity_score']])

        matched_control = control.iloc[indices.flatten()]
        matched_df = pd.concat([treated.reset_index(drop=True), matched_control.reset_index(drop=True)], axis=0)
        return matched_control, matched_df