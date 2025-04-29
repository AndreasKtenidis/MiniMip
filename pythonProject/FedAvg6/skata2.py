import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# 1. Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# 2. Define the Logistic Regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def federated_training(self, train_loader2):
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        for epoch in range(10):
            self.train()
            for train_batch_X, train_batch_y in train_loader2:
                optimizer.zero_grad()
                train_outputs = self(train_batch_X)
                loss = criterion(train_outputs, train_batch_y)
                tensor = torch.tensor(1)
                print(loss)
                loss.backward()
                optimizer.step()

model = LogisticRegression(input_dim=20)

model.federated_training(train_loader)
# # 3. Define loss function and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # 4. Training loop
# for epoch in range(10):
#     model.train()
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()

# 5. Evaluation (Plot model outputs)
model.eval()

# Collect probabilities and predictions for plotting
all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        probs = outputs.squeeze().numpy()  # Get probabilities
        preds = (outputs > 0.5).float().squeeze().numpy()  # Get binary predictions
        labels = batch_y.squeeze().numpy()  # True labels

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

# 6. Plotting
plt.figure(figsize=(10, 6))

# Plot probabilities vs. true labels
plt.subplot(1, 2, 1)
plt.scatter(range(len(all_probs)), all_probs, c=all_labels, cmap='coolwarm', label='Predicted Probabilities', alpha=0.6)
lala = all_preds==all_labels


matches = (np.array(all_preds) == np.array(all_labels)).sum()/len(all_labels)
print(matches)

plt.title('Predicted Probabilities vs. True Labels')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Probability')
plt.colorbar(label='True Label')

# Plot binary predictions vs. true labels
plt.subplot(1, 2, 2)
plt.scatter(range(len(all_preds)), all_preds, c=all_labels, cmap='coolwarm', label='Binary Predictions', alpha=0.6)
plt.title('Binary Predictions vs. True Labels')
plt.xlabel('Sample Index')
plt.ylabel('Binary Prediction')
plt.colorbar(label='True Label')

plt.tight_layout()
plt.show()
