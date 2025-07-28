import pandas as pd
import requests
from io import StringIO
from linearmodels.panel import RandomEffects

# Load Grunfeld data
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/plm/Grunfeld.csv"
data = pd.read_csv(url)

# Verify columns
print("Available columns:", data.columns.tolist())  # Should show: ['index', 'rownames', 'inv', 'value', 'capital']

# Clean and prepare data
data = data.rename(columns={'inv': 'invest'})  # Rename 'inv' to 'invest' for clarity
data = data.set_index(['firm', 'year'])  # Assuming 'firm' and 'year' are in the data

# Run model with correct column names
model = RandomEffects(
    data['invest'],  # Dependent variable (formerly 'inv')
    data[['value', 'capital']]  # Predictors
).fit(cov_type='robust')

print(model)

print(model.params)  # Returns a pandas Series

# intercept = model.params['const']       # Intercept term
# value_coef = model.params['value']     # Coefficient for 'value'
# capital_coef = model.params['capital'] # Coefficient for 'capital'
