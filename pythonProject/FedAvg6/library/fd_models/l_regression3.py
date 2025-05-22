import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.linear_model import LinearRegression as SkLinearRegression


class InsuranceDataset(Dataset):
    """PyTorch Dataset for insurance data"""

    def __init__(self, features, targets):
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class InsuranceLinearRegression:
    """End-to-end linear regression pipeline for insurance data"""

    def __init__(self):
        self.feature_preprocessor = None
        self.target_scaler = None
        self.model = None
        self.sk_model = SkLinearRegression()

    def _preprocess_data(self, x, y=None, fit=False):
        """Handle feature preprocessing and target scaling"""
        # Feature preprocessing
        numeric_features = ['age', 'bmi', 'children']
        categorical_features = ['sex', 'smoker', 'region']

        if fit:
            self.feature_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ])
            x_processed = self.feature_preprocessor.fit_transform(x)
        else:
            x_processed = self.feature_preprocessor.transform(x)

        # Target scaling
        if y is not None:
            if fit:
                self.target_scaler = StandardScaler()
                y_processed = self.target_scaler.fit_transform(y)
            else:
                y_processed = self.target_scaler.transform(y)
            return x_processed.astype(np.float32), y_processed.astype(np.float32)

        return x_processed.astype(np.float32)

    def prepare_data(self, test_size=0.2, random_state=42):
        """Load and split the dataset"""
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        df = pd.read_csv(url)

        x = df.drop('charges', axis=1)
        y = df['charges'].values.reshape(-1, 1)




        # Preprocess and split
        x_processed, y_processed = self._preprocess_data(x, y, fit=True)
        x_train, x_test, y_train, y_test = train_test_split(
            x_processed, y_processed, test_size=test_size, random_state=random_state)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def create_dataloaders(x_train, x_test, y_train, y_test, batch_size=32):
        """Create PyTorch dataloaders"""
        train_dataset = InsuranceDataset(x_train, y_train)
        test_dataset = InsuranceDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train_pytorch_model(self, train_loader, input_size, learning_rate=0.001, epochs=1000):
        """Train PyTorch linear regression model"""
        self.model = nn.Linear(input_size, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    def evaluate(self, test_loader, x_test, y_test):
        """Evaluate both models"""
        # PyTorch evaluation
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                test_loss += nn.MSELoss()(outputs, targets).item()

        scaled_test_mse = test_loss / len(test_loader)
        test_mse = self.target_scaler.inverse_transform([[np.sqrt(scaled_test_mse)]]) ** 2
        print(f'PyTorch Test MSE: {test_mse[0][0]:.2f}')

        # Scikit-learn evaluation
        sk_r2 = self.sk_model.score(x_test, y_test)
        print(f'Scikit-learn R²: {sk_r2:.4f}')

    def predict(self, input_data):
        """Make predictions with both models"""
        # Preprocess input
        processed_input = self._preprocess_data(pd.DataFrame([input_data]))
        tensor_input = torch.from_numpy(processed_input).float()

        # PyTorch prediction
        with torch.no_grad():
            pytorch_pred = self.model(tensor_input)
        pytorch_charge = self.target_scaler.inverse_transform(pytorch_pred.numpy())

        # Scikit-learn prediction
        sklearn_charge = self.target_scaler.inverse_transform(
            self.sk_model.predict(processed_input).reshape(-1, 1))

        print(f'PyTorch prediction: ${pytorch_charge[0][0]:.2f}')
        print(f'Scikit-learn prediction: ${sklearn_charge[0][0]:.2f}')


# Usage example
if __name__ == "__main__":
    insurance_model = InsuranceLinearRegression()


    # Prepare data
    X_train, X_test, y_train2, y_test2 = insurance_model.prepare_data()
    train_loader2, test_loader2 = insurance_model.create_dataloaders(X_train, X_test, y_train2, y_test2)

    # Train models
    insurance_model.train_pytorch_model(train_loader2, input_size=X_train.shape[1])

    # Evaluate
    insurance_model.evaluate(test_loader2, X_test, y_test2)

    # Predict
    sample_input = {
        'age': 40,
        'sex': 'male',
        'bmi': 30.0,
        'children': 2,
        'smoker': 'yes',
        'region': 'southeast'
    }
    insurance_model.predict(sample_input)