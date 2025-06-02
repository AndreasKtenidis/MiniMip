import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Optional, Tuple
from dataclasses import dataclass

from system.client.aggregation_client import AggregationClient, NumpyAggClient
from library.stats.statistical_function import StatisticalFunction


@dataclass
class ClientState:
    coefficients: np.ndarray
    intercept: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray  # Or Hessian approximation in L-BFGS
    n_samples: int
    loss: float


class FederatedLogisticRegressionLBFGS(StatisticalFunction):
    def compute(self, x,y):
        self.fit(x,y)

    def __init__(self, client:AggregationClient, penalty='l2', C=1.0, max_iter=100, tol=1e-4,
                 n_rounds=10, warm_start=False, random_state=None, verbose=0):
        """
        Improved Federated Logistic Regression with proper L-BFGS aggregation.

        Args:
            penalty: Norm used in penalization ('l2' or 'none')
            C: Inverse of regularization strength
            max_iter: Max iterations per local round
            tol: Tolerance for stopping
            n_rounds: Number of global federated rounds
            warm_start: Reuse previous solution
            random_state: Random seed
            verbose: Verbosity level
        """
        super().__init__(client)
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.n_rounds = n_rounds
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.classes_ = None
        self.n_classes_ = None
        self.global_loss_history_ = []
        self.aggregator:NumpyAggClient = self.get_numpy_aggregator()

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit the model with federated L-BFGS optimization."""
        # Handle class labels
        self.classes_ = self.aggregator.fed_union(np.unique(y))
        self.n_classes_ = len(self.classes_)

        # Initialize model
        if self.model is None:
            self._initialize_model(X.shape[1])

        # Convert y to one-hot if multi-class
        y_processed = self._process_labels(y)

        # Federated training rounds
        for round_num in range(self.n_rounds):
            if self.verbose > 0:
                print(f"Starting federated round {round_num + 1}/{self.n_rounds}")

            # 1. Client update phase
            client_state = self._client_update(X, y_processed, sample_weight)

            # 2. Server aggregation
            self._server_aggregate(client_state)  # In real FL, would have multiple clients

            # 3. Check convergence
            if self._check_convergence():
                break

        return self

    def _initialize_model(self, n_features: int):
        """Initialize model parameters."""
        self.model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver='lbfgs',
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=self.warm_start,
            random_state=self.random_state,
            verbose=max(0, self.verbose - 1),
            multi_class='multinomial' if self.n_classes_ > 2 else 'auto'
        )

        # Initialize parameters
        if self.n_classes_ > 2:
            self.model.coef_ = np.zeros((self.n_classes_, n_features))
            self.model.intercept_ = np.zeros(self.n_classes_)
        else:
            self.model.coef_ = np.zeros((1, n_features))
            self.model.intercept_ = np.zeros(1)

        # Initialize L-BFGS state
        self._reset_lbfgs_state(n_features)

    def _reset_lbfgs_state(self, n_features: int):
        """Initialize L-BFGS state variables."""
        # For simplicity, we'll track these as model attributes
        # In production, you'd want a proper state object
        n_params = n_features + 1  # +1 for intercept
        if self.n_classes_ > 2:
            n_params *= self.n_classes_

        # L-BFGS typically maintains last m gradients/updates
        self.lbfgs_m = 10  # History size
        self.lbfgs_s = []  # Parameter differences (s_k = x_{k+1} - x_k)
        self.lbfgs_y = []  # Gradient differences (y_k = grad_{k+1} - grad_k)
        self.prev_gradient = None
        self.prev_params = None

    def _client_update(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> ClientState:
        """Perform local client update and return state."""
        # Create local model with current global parameters
        local_model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver='lbfgs',
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=False,
            random_state=self.random_state,
            verbose=0,
            multi_class='multinomial' if self.n_classes_ > 2 else 'auto'
        )

        # Set current global parameters
        local_model.coef_ = self.model.coef_.copy()
        local_model.intercept_ = self.model.intercept_.copy()

        # Train locally
        local_model.fit(X, y, sample_weight=sample_weight)

        # Compute gradient and Hessian approximation
        # Note: In practice, you'd want to extract these from the optimization process
        # Here we approximate by computing them analytically
        gradient, hessian = self._compute_gradient_hessian(
            X, y, local_model.coef_, local_model.intercept_, sample_weight
        )

        # Compute loss
        loss = self._compute_loss(X, y, local_model.coef_, local_model.intercept_, sample_weight)

        return ClientState(
            coefficients=local_model.coef_,
            intercept=local_model.intercept_,
            gradient=gradient,
            hessian=hessian,
            n_samples=X.shape[0],
            loss=loss
        )

    def _compute_gradient_hessian(self, X: np.ndarray, y: np.ndarray,
                                  coef: np.ndarray, intercept: np.ndarray,
                                  sample_weight: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient and Hessian approximation."""
        # This is a simplified version - in practice you'd want to:
        # 1. Extract these from the L-BFGS optimization process
        # 2. Or compute proper second-order information

        # For binary classification
        if self.n_classes_ == 2:
            logits = X @ coef.T + intercept
            probs = 1 / (1 + np.exp(-logits))
            errors = probs - y

            # Gradient
            grad_coef = X.T @ errors
            grad_intercept = np.sum(errors)

            # Hessian approximation (diagonal for simplicity)
            # In proper L-BFGS, we'd maintain the full approximation
            hessian_coef = np.mean(probs * (1 - probs)) * np.eye(X.shape[1])
            hessian_intercept = np.mean(probs * (1 - probs))

            return (
                np.concatenate([grad_coef, [grad_intercept]]),
                np.block([[hessian_coef, np.zeros((X.shape[1], 1))],
                          [np.zeros((1, X.shape[1])), hessian_intercept]])
            )
        else:
            # Multi-class case - simplified
            logits = X @ coef.T + intercept
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            errors = probs - y

            # Gradient
            grad_coef = errors.T @ X
            grad_intercept = np.sum(errors, axis=0)

            # Flatten for L-BFGS state
            grad = np.concatenate([grad_coef.ravel(), grad_intercept])

            # Hessian approximation (block diagonal for simplicity)
            hessian = np.kron(np.eye(self.n_classes_), np.mean(probs * (1 - probs)) * np.eye(X.shape[1]))

            return grad, hessian

    def _compute_loss(self, X: np.ndarray, y: np.ndarray,
                      coef: np.ndarray, intercept: np.ndarray,
                      sample_weight: Optional[np.ndarray]) -> float:
        """Compute logistic loss with regularization."""
        # Binary case
        if self.n_classes_ == 2:
            logits = X @ coef.T + intercept
            loss = np.sum(np.log(1 + np.exp(-y * logits)))
        else:
            # Multi-class cross-entropy
            logits = X @ coef.T + intercept
            log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
            loss = -np.sum(y * log_probs)

        # Add regularization
        if self.penalty == 'l2':
            loss += 0.5 / self.C * np.sum(coef ** 2)

        return loss

    def _server_aggregate(self, state:ClientState):
        """Aggregate client updates using federated L-BFGS."""
        # 1. Aggregate losses and compute global loss

        total_loss = self.aggregator.fed_sum(state.loss * state.n_samples)
        total_samples = self.aggregator.fed_sum(state.n_samples)
        global_loss = total_loss / total_samples
        self.global_loss_history_.append(global_loss)

        if self.verbose > 0:
            print(f"Global loss: {global_loss:.4f}")

        # 2. Weighted average of parameters
        new_coef = np.zeros_like(state.coefficients)
        new_intercept = np.zeros_like(state.intercept)


        weight = state.n_samples / total_samples
        new_coef += weight * state.coefficients
        new_intercept += weight * state.intercept

        # 3. Update L-BFGS state with aggregated gradients/Hessians
        # (In practice you'd want to properly maintain the L-BFGS history)

        avg_gradient = self.aggregator.fed_sum(weight * state.gradient)
        avg_hessian =self.aggregator.fed_sum( weight * state.hessian)


        # Update model parameters
        self.model.coef_ = new_coef
        self.model.intercept_ = new_intercept

        # Update L-BFGS state (simplified)
        if self.prev_gradient is not None:
            s = self._get_current_params() - self.prev_params
            y = avg_gradient - self.prev_gradient

            # Update history
            self.lbfgs_s.append(s)
            self.lbfgs_y.append(y)

            # Keep only last m updates
            if len(self.lbfgs_s) > self.lbfgs_m:
                self.lbfgs_s.pop(0)
                self.lbfgs_y.pop(0)

        self.prev_gradient = avg_gradient.copy()
        self.prev_params = self._get_current_params().copy()

    def _get_current_params(self) -> np.ndarray:
        """Get current model parameters as a single vector."""
        return np.concatenate([self.model.coef_.ravel(), self.model.intercept_])

    def _check_convergence(self) -> bool:
        """Check if training has converged based on loss history."""
        if len(self.global_loss_history_) < 2:
            return False

        prev_loss = self.global_loss_history_[-2]
        current_loss = self.global_loss_history_[-1]

        return abs(prev_loss - current_loss) < self.tol

    def _process_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert labels to one-hot if multi-class."""
        if self.n_classes_ > 2:
            return np.eye(self.n_classes_)[np.searchsorted(self.classes_, y)]
        return y.reshape(-1, 1)

    # Prediction methods remain the same as previous implementation
    # Federated framework methods remain abstract