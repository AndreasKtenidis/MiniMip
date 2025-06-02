import numpy as np

from library.stat_models.statistical_Model import StatisticalModel
import torch
import torch.nn as nn
import torch.optim as optim


class FederatedLinearRegression(StatisticalModel):

    def __init__(self, client):
        super().__init__(client)
        self.model = None
        self.aggregator = self.get_numpy_aggregator()

    def fit(self, x, y):
        self.model = nn.Linear(x.shape[1], 1, bias=True)
        # TODO this step could be avoided if the server was sending the params at first step
        params = self._get_model_params()
        params2 = self.aggregator.fed_avg(params)
        self._set_model_params(params2)
        # End TODO
        self._train(x, y)

    def _get_model_params(self) -> np.ndarray:
        """Flatten and concatenate model weights and bias to numpy."""
        weights = self.model.weight.detach().cpu().numpy().flatten()
        bias = self.model.bias.detach().cpu().numpy()
        return np.concatenate([weights, bias])

    def _set_model_params(self, flat_params: np.ndarray):
        """Update model weights and bias from flattened numpy array."""
        with torch.no_grad():
            weight_shape = self.model.weight.shape
            bias_shape = self.model.bias.shape

            weight_size = weight_shape.numel()
            weights = flat_params[:weight_size].reshape(weight_shape)
            bias = flat_params[weight_size:].reshape(bias_shape)

            self.model.weight.data = torch.tensor(weights, dtype=torch.float32)
            self.model.bias.data = torch.tensor(bias, dtype=torch.float32)

    def _train(self, x: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int =500):
        """Train the model locally and apply fed_avg after each epoch."""

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        local_params = self._get_model_params()
        for epoch in range(epochs):
            self.model.fit()
            optimizer.zero_grad()
            outputs = self.model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            # Perform federated averaging
            local_params = self._get_model_params()
            global_params = self.aggregator.fed_avg(local_params)
            self._set_model_params(global_params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for input data."""
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(x_tensor).numpy().flatten()
        return predictions
