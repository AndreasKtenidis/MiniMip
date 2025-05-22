import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Load the dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

# Preprocess the data
X = df.drop('charges', axis=1)
y = df['charges'].values.reshape(-1, 1)

# Define preprocessing
numeric_features = ['age', 'bmi', 'children']
numeric_transformer = StandardScaler()

categorical_features = ['sex', 'smoker', 'region']
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Transform features and target
X_processed = preprocessor.fit_transform(X).astype(np.float32)
y_scaler = StandardScaler()
y_processed = y_scaler.fit_transform(y).astype(np.float32)  # Scale target!

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, test_size=0.2, random_state=42
)


# PyTorch Dataset
class InsuranceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = InsuranceDataset(X_train, y_train)
test_dataset = InsuranceDataset(X_test, y_test)

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Initialize model
input_size = X_train.shape[1]
model = LinearRegression(input_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate

# Training loop
num_epochs = 1000
train_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}')

# Evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()

scaled_test_mse = test_loss / len(test_loader)
# Convert back to original scale
test_mse = y_scaler.inverse_transform([[np.sqrt(scaled_test_mse)]]) ** 2
print(f'Test MSE (original scale): {test_mse[0][0]:.2f}')

# Example prediction
sample_input = torch.tensor([[40, 30.0, 2, 1, 1, 0, 0, 1]], dtype=torch.float32)  # Processed features
predicted_charge = y_scaler.inverse_transform(model(sample_input).detach().numpy())
print(f'Predicted insurance charge: ${predicted_charge[0][0]:.2f}')
def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)

# Usage:
y_test_tensor = torch.from_numpy(y_test)
with torch.no_grad():
    y_pred = model(torch.from_numpy(X_test).numpy())
print(f'PyTorch R²: {r2_score(y_test, y_pred):.4f}')

# Compare with sklearn
from sklearn.linear_model import LinearRegression as SkLinearRegression

sk_model = SkLinearRegression()
sk_model.fit(X_train, y_train)
print(f'Sklearn Test R²: {sk_model.score(X_test, y_test):.4f}')

