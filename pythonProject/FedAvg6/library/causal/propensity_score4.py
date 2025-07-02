import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

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


class PropensityScore():

    def compute(self, X: np.array, y: np.array):
        model = LogisticRegression()
        model.fit(X, y)
        propensity_scores = model.predict_proba(X)[:, 1]  # Probability of treatment
        # Add propensity scores to dataframe
        data['propensity_score'] = propensity_scores

        print("Propensity Scores (scikit-learn):")
        print(data[['ID', 'Treated', 'propensity_score']])

        # Step 2: Separate treated and control units
        treated = data[data['Treated'] == 1].copy()
        control = data[data['Treated'] == 0].copy()

        # Step 3: Nearest neighbor matching based on propensity score
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[['propensity_score']])
        distances, indices = nbrs.kneighbors(treated[['propensity_score']])

        # Get matched control indices
        matched_control_indices = control.iloc[indices.flatten()].index

        # Step 4: Combine matched pairs
        matched_pairs = pd.DataFrame({
            'Treated_ID': treated['ID'].values,
            'Control_ID': control.loc[matched_control_indices, 'ID'].values,
            'Treated_Cholesterol': treated['Cholesterol'].values,
            'Control_Cholesterol': control.loc[matched_control_indices, 'Cholesterol'].values
        })

        print("\nMatched Pairs:")
        print(matched_pairs)

        # Step 5: Calculate Average Treatment Effect on the Treated (ATT)
        matched_pairs['Cholesterol_Diff'] = matched_pairs['Treated_Cholesterol'] - matched_pairs['Control_Cholesterol']
        ATT = matched_pairs['Cholesterol_Diff'].mean()

        print(f"\nEstimated ATT (Treatment Effect on Cholesterol): {ATT:.2f}")


propScore = PropensityScore()
propScore.compute(X2, y2)
