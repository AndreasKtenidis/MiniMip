import pandas as pd
import statsmodels.api as sm

# Step 1: Create the data
data = pd.DataFrame({
    'Patient': [1, 2, 3, 4, 5, 6],
    'Age': [70, 60, 50, 40, 55, 65],
    'Treatment': [1, 1, 0, 0, 1, 0],  # More variation
    'Outcome': [130, 125, 140, 135, 128, 138]
})

# Step 2: Fit logistic regression to estimate propensity scores
# We'll use Age as the confounder to predict Treatment
X = sm.add_constant(data['Age'])  # add intercept
y = data['Treatment']

logit = sm.Logit(y, X)
result = logit.fit(disp=False)

# Predict propensity scores
data['Propensity'] = result.predict(X)

# Step 3: Calculate IPTW weights
data['Weight'] = data.apply(
    lambda row: 1/row['Propensity'] if row['Treatment']==1 else 1/(1-row['Propensity']),
    axis=1
)

# Step 4: Calculate weighted treatment effect
treated = data.loc[data['Treatment']==1]
untreated = data.loc[data['Treatment']==0]

treated_mean = (treated['Outcome'] * treated['Weight']).sum() / treated['Weight'].sum()
untreated_mean = (untreated['Outcome'] * untreated['Weight']).sum() / untreated['Weight'].sum()

treatment_effect = treated_mean - untreated_mean

print("Data with Propensity Scores and Weights:")
print(data)
print("\nWeighted Treated Mean:", treated_mean)
print("Weighted Untreated Mean:", untreated_mean)
print("Estimated Treatment Effect (IPTW):", treatment_effect)
