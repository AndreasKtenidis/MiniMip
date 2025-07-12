import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load your Titanic dataset
# -------------------------------

# For illustration, here we use seaborn's Titanic dataset
import seaborn as sns



def ipwt_non_federated(df):
    print("\nOriginal columns:")
    print(df.columns)
    # Use columns:
    # 'pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare'
    df = df[['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
    # Treatment: sex (female=1, male=0)
    df['Treatment'] = LabelEncoder().fit_transform(df['sex'])  # female=1, male=0
    # Outcome: survived
    df['Outcome'] = df['survived']
    # Confounders
    confounders = ['pclass', 'age', 'sibsp', 'parch', 'fare']
        # -------------------------------
    # 2. Estimate propensity scores: P(Treatment | Confounders)
    # -------------------------------
    X = df[confounders]
    y = df['Treatment']
    logistic = LogisticRegression(max_iter=200)
    logistic.fit(X, y)
    df['ps'] = logistic.predict_proba(X)[:, 1]

    # -------------------------------
    # 3. Compute IPTW weights
    # -------------------------------
    df['weight'] = np.where(
        df['Treatment'] == 1,
        1 / df['ps'],
        1 / (1 - df['ps'])
    )
    print(df)
    return df

df = sns.load_dataset('titanic')
ipwt_non_federated(df)