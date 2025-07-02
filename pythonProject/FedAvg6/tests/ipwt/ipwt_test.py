import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression

# Load diabetes dataset
df = get_dataframe()

# Covariates to model treatment assignment (propensity)
X = df.drop(columns=['treatment', 'outcome'])
T = df['treatment']

# 1. Estimate propensity scores P(T=1 | X)
propensity_model = LogisticRegression(max_iter=1000)
propensity_model.fit(X, T)
df['propensity_score'] = propensity_model.predict_proba(X)[:, 1]

# 2. Calculate IPW weights
df['weight'] = np.where(
    df['treatment'] == 1,
    1 / df['propensity_score'],
    1 / (1 - df['propensity_score'])
)

# 3. Estimate weighted means for treated and control
treated = df[df['treatment'] == 1]
control = df[df['treatment'] == 0]

ate_ipw = (
    (treated['outcome'] * treated['weight']).sum() / treated['weight'].sum()
    -
    (control['outcome'] * control['weight']).sum() / control['weight'].sum()
)

print(f"Estimated Average Treatment Effect (ATE) by IPW on Diabetes dataset: {ate_ipw:.2f}")

def get_dataframe():
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame

    # For demonstration, create a binary treatment variable:
    # e.g., treatment = 1 if BMI above median, else 0
    print(df['bmi'].median())
    df['treatment'] = (df['bmi'] > -0.007283766209687899).astype(int)

    print(df['bmi'])
    print(df['treatment'])

    # Outcome: let's use the 'target' variable (disease progression)
    df['outcome'] = df['target']