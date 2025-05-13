import pandas as pd
import numpy as np
from urllib.request import urlopen

# Load Titanic dataset from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 1: Build contingency table (Survived vs Pclass)
# Survived: 0 = Died, 1 = Survived
contingency_table = pd.crosstab(df['Pclass'], df['Survived'])
print(contingency_table)

print("Contingency Table:\n", contingency_table, "\n")

# Step 2: Implement Chi-squared test from scratch
def chi_squared_test(observed):
    observed = np.array(observed)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    expected = row_totals @ col_totals / grand_total
    chi2_stat = ((observed - expected) ** 2 / expected).sum()
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    return chi2_stat, dof, expected

chi2, dof, expected = chi_squared_test(contingency_table.values)

print("Chi-squared Statistic:", round(chi2, 4))
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", np.round(expected, 2))
print(contingency_table)

print('\n')
print(df['Pclass'])
print('---->',df['Survived'])
print('?????',pd.crosstab(df['Pclass'], df['Survived']))