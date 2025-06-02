import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mord import LogisticIT



# Load Wine Quality dataset (red wine)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')



# Inspect
print(data.head())

# Features and target
X = (data.drop(columns=['quality']).values)
y = (data['quality'].values) # quality is ordinal: 3-8 (integers)

# Because the dataset is imbalanced and for simplicity,
# we will only use quality scores 3 to 7 and discard 8.
mask = y <= 7
X = X[mask]
y = y[mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit ordinal logistic regression model
model = LogisticIT()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
