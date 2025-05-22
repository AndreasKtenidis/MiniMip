import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
data = {
    "age": [25, 30, 35],
    "color": ["red", "blue", "green"],  # Categorical (nominal)
    "size": ["S", "M", "L"],           # Categorical (ordinal)
    "income": [50000, 80000, 120000],  # Numerical
    "target": [0, 1, 0]                # Target variable
}
df = pd.DataFrame(data)
X = df.drop("target", axis=1)
y = df["target"]

preprocessor = ColumnTransformer(
    transformers=[
        # ("onehot", OneHotEncoder(), ["color"]),     # Nominal
        ("ordinal", OrdinalEncoder(), ["size"]),   # Ordinal
        # ("scaler", StandardScaler(), ["age", "income"])  # Numerical
    ],
    remainder="passthrough"  # Keeps unprocessed columns (if any)
)
# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier", RandomForestClassifier())  # Example model
# ])
test = preprocessor.fit_transform(X)
type(test)
print(X)
print(test)
# predictions = pipeline.predict(X)

# print(X)

