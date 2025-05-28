import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Simulated data
np.random.seed(42)
n = 1000
age = np.random.normal(50, 10, n)
comorbidity = np.random.binomial(1, 0.3, n)

# Treatment assignment (depends on age and comorbidity)
p_treatment = 1 / (1 + np.exp(-(-2 + 0.05*age + 1.5*comorbidity)))
treatment = np.random.binomial(1, p_treatment)

# Outcome (affected by treatment and confounders)
outcome = 5 + 3*treatment - 0.1*age + 2*comorbidity + np.random.normal(0, 1, n)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'comorbidity': comorbidity,
    'treatment': treatment,
    'outcome': outcome
})

model = LogisticRegression()
model.fit(df[['age', 'comorbidity']], df['treatment'])
propensity_scores = model.predict_proba(df[['age', 'comorbidity']])[:,1]
df['ps'] = propensity_scores
print(df['ps'])