import numpy as np
from sklearn.linear_model import LogisticRegression

from pythonProject.FedAvg6.library.templates.statistical_model import StatisticalModel
from pythonProject.FedAvg6.system.client.grpc_agg_client import GRPCClient

class FederatedLogisticRegressionClientSaSo(StatisticalModel):

    def __init__(self, client: GRPCClient,  model_params=None):

        super().__init__(client)
        self.agg = self.get_numpy_aggregator()
        self.model_params = model_params or {
            'solver': 'saga', 'penalty': 'l2', 'fit_intercept': True,
            'max_iter': 100, 'warm_start': True
        }
        self.model = LogisticRegression(**self.model_params)



    def get_weights(self):
        """Return model weights (coef_ and intercept_) as flattened array."""
        coef = self.model.coef_.flatten()
        intercept = self.model.intercept_
        return np.concatenate([coef, intercept])

    def set_weights(self, weights: np.ndarray):
        """Set model weights from a flattened array."""
        n_features = self.x_shape
        coef = weights[:n_features].reshape(1, -1)
        intercept = weights[n_features:]
        self.model.coef_ = coef
        self.model.intercept_ = intercept



    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 100):
        self.x_shape = X.shape[1]
        """
        Federated training loop. Performs one epoch of local training followed
        by federated aggregation after each epoch.

        Args:
            num_epochs (int): Number of global training epochs (rounds of aggregation).
        """
        n_samples = X.shape[0]

        for epoch in range(num_epochs):
            print(f"[Client] Federated Epoch {epoch + 1}/{num_epochs}")

            # One local epoch: partial_fit for better control
            if hasattr(self.model, 'partial_fit'):
                # For binary classification, need to specify classes on first call
                if epoch == 0:
                    self.model.partial_fit(X, y, classes=np.unique(y))
                else:
                    self.model.partial_fit(X, y)
            else:
                # Fallback to full fit, warm_start avoids reinitializing weights
                self.model.fit(X, y)

            # Extract weights (coef_ and intercept_)
            local_weights = self.get_weights()

            # Federated weighted average
            avg_weights = self.agg.fed_weighted_avg(local_weights, weight=n_samples)

            # Update local model weights to the aggregated version
            self.set_weights(avg_weights)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)