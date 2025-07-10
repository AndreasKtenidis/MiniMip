import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mord import LogisticIT
from scipy.stats import chi2_contingency



# Load Wine Quality dataset (red wine)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)



def chi_test(df):
    # Inspect
    print(df.head())

    # Features and target
    # Create a contingency table
    contingency_table = pd.crosstab(df['Pclass'],df['Survived'])

    print("Contingency Table:")
    print(contingency_table)

    # Run Chi-Squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\nChi-squared Statistic:", chi2)
    print("p-value:", p)
    print("Degrees of Freedom:", dof)

def fisher_test(df):
    table = pd.crosstab(df['Sex'], df['Survived'])
    print("Contingency Table:")
    print(table)

    from scipy.stats import fisher_exact

    oddsratio, p_value = fisher_exact(table)

    print("\nOdds Ratio:", oddsratio)
    print("p-value:", p_value)

chi_test(df)
print('---------------------')
fisher_test(df)

