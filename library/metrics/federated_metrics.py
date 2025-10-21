import numpy as np
from library.utils.aggregation_client import AggregationClientInterface
from library.utils.numpy_aggregator import NumpyAggregator


class LogisticRegressionFedMetrics:

    def __init__(self, client: AggregationClientInterface):
        self.numpy_aggregator = NumpyAggregator(client)

    def accuracy_score(self, *, y_true:np.ndarray, y_pred:np.ndarray):
        total_correct = self.numpy_aggregator.global_sum(y_true == y_pred)
        total_samples = self.numpy_aggregator.global_count(y_true)
        return total_correct / total_samples

    def precision_score(self, *, y_true: np.ndarray, y_pred: np.ndarray):
        # True Positives
        tp = self.numpy_aggregator.global_sum((y_pred == 1) & (y_true == 1))
        # Predicted Positives
        predicted_positives = self.numpy_aggregator.global_sum(y_pred == 1)
        # Avoid division by zero
        precision = tp / predicted_positives if predicted_positives > 0 else 0.0
        return precision

    def recall_score(self, *, y_true: np.ndarray, y_pred: np.ndarray):
        # True Positives
        tp = self.numpy_aggregator.global_sum((y_pred == 1) & (y_true == 1))
        # Actual Positives
        actual_positives = self.numpy_aggregator.global_sum(y_true == 1)
        # Avoid division by zero
        recall = tp / actual_positives if actual_positives > 0 else 0.0
        return recall

    def f1_score(self,*, y_true: np.ndarray, y_pred: np.ndarray):
        precision = self.precision_score(y_true=y_true, y_pred= y_pred)
        recall = self.recall_score(y_true=y_true, y_pred= y_pred)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def log_loss(self,*, y_true: np.ndarray, y_prob: np.ndarray, eps=1e-15):
        # Clip probabilities for numerical stability
        y_prob = np.clip(y_prob, eps, 1 - eps)
        # Compute per-sample log loss
        per_sample_loss = - (y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

        total_loss = self.numpy_aggregator.global_sum(per_sample_loss)
        total_samples = self.numpy_aggregator.global_count(y_true)
        return total_loss / total_samples

    def auc_score(self, *, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Compute federated AUC by aggregating TP/FP counts over thresholds.
        Args:
            y_true: np.ndarray - true binary labels for this client.
            y_prob: np.ndarray - predicted probabilities for this client.

        Returns:
            float - the global AUC score.
        """

        # 1. Get unique thresholds from local predicted probabilities
        local_thresholds = np.unique(y_prob)

        # 2. Federate union of thresholds across all clients
        global_thresholds = self.numpy_aggregator.fed_union(local_thresholds)
        global_thresholds = np.sort(global_thresholds)

        # 3. Compute local TP and FP for each threshold
        local_tps = np.array([np.sum((y_prob >= thr) & (y_true == 1)) for thr in global_thresholds])
        local_fps = np.array([np.sum((y_prob >= thr) & (y_true == 0)) for thr in global_thresholds])

        # 4. Federated sum of TP and FP counts
        global_tps = self.numpy_aggregator.fed_sum(local_tps)
        global_fps = self.numpy_aggregator.fed_sum(local_fps)

        # 5. Get total positive and negative sample counts
        local_pos_count = np.sum(y_true == 1)
        local_neg_count = np.sum(y_true == 0)

        global_pos_count = self.numpy_aggregator.fed_sum(np.array([local_pos_count]))[0]
        global_neg_count = self.numpy_aggregator.fed_sum(np.array([local_neg_count]))[0]

        # 6. Compute TPR and FPR arrays (avoid division by zero)
        tpr = global_tps / global_pos_count if global_pos_count > 0 else np.zeros_like(global_tps)
        fpr = global_fps / global_neg_count if global_neg_count > 0 else np.zeros_like(global_fps)

        # 7. Sort by FPR for integration
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # 8. Compute AUC using trapezoidal rule
        auc = np.trapz(tpr_sorted, fpr_sorted)
        return auc


class LinearRegressionFedMetrics:

    def __init__(self, client: AggregationClientInterface):
        self.numpy_aggregator = NumpyAggregator(client)

    def mean_squared_error(self,*, y_true:np.ndarray, y_pred:np.ndarray):
        mse = self.numpy_aggregator.global_avg((y_true - y_pred) ** 2)
        return mse

    def mean_absolute_error(self,*, y_true:np.ndarray, y_pred:np.ndarray):
        mae = self.numpy_aggregator.global_avg(np.abs(y_true - y_pred))
        return mae

    def r2_score(self,*, y_true, y_pred):
        """
        Compute R² Score (coefficient of determination) between true and predicted values.

        Parameters:
            y_true (array-like): True target values
            y_pred (array-like): Predicted values

        Returns:
            float: R² Score
        """
        ss_res = self.numpy_aggregator.global_sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = self.numpy_aggregator.global_sum((y_true - self.numpy_aggregator.fed_avg(y_true)) ** 2)  # Total sum of squares
        #
        return 1 - (ss_res / ss_tot)
