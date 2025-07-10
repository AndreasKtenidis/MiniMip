import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the insurance dataset (replace with your file path)
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
insurance_df = pd.read_csv(url)

# Display first few rows
print(insurance_df.head())

# Convert categorical variables (sex, smoker, region) into numerical using one-hot encoding
insurance_processed = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Define features (X) and target (y)
X = insurance_processed.drop('charges', axis=1)  # Features
y = insurance_processed['charges']              # Target (insurance cost)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")