import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from library.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.templates.statistical_function import StatisticalFunction

# Sample data
data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'Treated': [1, 1, 1, 0, 0, 0],
    'Age': [30, 40, 35, 32, 45, 37],
    'BMI': [25, 28, 27, 26, 30, 27],
    'Cholesterol': [180, 175, 170, 185, 190, 182]
})

# Step 1: Estimate propensity scores using scikit-learn logistic regression
X2 = data[['Age', 'BMI']].values
y2 = data['Treated'].values


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