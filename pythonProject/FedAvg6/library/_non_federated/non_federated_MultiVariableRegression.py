import pandas as pd
import statsmodels.api as sm

# Example data
data = {
    'BloodPressure': [120, 130, 125, 140, 135, 128, 118, 145, 138, 132],
    'ExerciseHours': [5, 3, 4, 2, 1, 4, 6, 2, 3, 5],
    'Age': [25, 45, 35, 50, 60, 40, 30, 55, 48, 38],
    'DietScore': [7, 5, 6, 4, 3, 6, 8, 3, 5, 7]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define dependent variable (outcome)
y = df['BloodPressure']

# Define independent variables (predictors) including all confounders
X = df[['ExerciseHours', 'Age', 'DietScore']]

# Add a constant term (intercept) to the predictors
X = sm.add_constant(X)

# Fit the multivariable regression model
model = sm.OLS(y, X).fit()


# Print the summary of the regression
print(model.summary())