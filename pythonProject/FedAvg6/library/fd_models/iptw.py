import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Δημιουργία τυχαίων δεδομένων
np.random.seed(42)
n = 1000
age = np.random.normal(50, 10, n)                  # ηλικία
comorbidity = np.random.binomial(1, 0.3, n)        # συννοσηρότητα (1 = ναι, 0 = όχι)

# Πιθανότητα να πάρει κάποιος θεραπεία (όχι τυχαία)
p_treatment = 1 / (1 + np.exp(-(-2 + 0.05*age + 1.5*comorbidity)))
treatment = np.random.binomial(1, p_treatment)

# Αποτέλεσμα (πίεση): επηρεάζεται από θεραπεία, ηλικία, συννοσηρότητα
outcome = 5 + 3*treatment - 0.1*age + 2*comorbidity + np.random.normal(0, 1, n)

# Φτιάχνουμε dataframe
df = pd.DataFrame({
    'age': age,
    'comorbidity': comorbidity,
    'treatment': treatment,
    'outcome': outcome
})



model = LogisticRegression()
model.fit(df[['age', 'comorbidity']], df['treatment'])

# Η πιθανότητα να πάρει θεραπεία με βάση τα χαρακτηριστικά
df['ps'] = model.predict_proba(df[['age', 'comorbidity']])[:,1]



df['iptw'] = np.where(df['treatment'] == 1,
                      1 / df['ps'],
                      1 / (1 - df['ps']))

# Weighted Least Squares (WLS) παλινδρόμηση
weighted_model = sm.WLS(df['outcome'], sm.add_constant(df['treatment']), weights=df['iptw']).fit()
print(weighted_model.summary())
