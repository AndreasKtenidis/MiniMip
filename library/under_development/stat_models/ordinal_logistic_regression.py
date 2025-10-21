# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from library.fd_models._ordinal_logistic_regression.fed_ordinal_regression import LogisticIT
# import numpy as np
# import inspect
#
#
# # Load Wine Quality dataset (red wine)
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# data = pd.read_csv(url, sep=';')
#
#
#
# # Inspect
# print(data.head())
#
# # Features and target
# X = data.drop(columns=['quality']).values
# y = data['quality'].values  # quality is ordinal: 3-8 (integers)
#
# # Because the dataset is imbalanced and for simplicity,
# # we will only use quality scores 3 to 7 and discard 8.
# mask = y <= 7
# X = X[mask]
# y = y[mask]
#
# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Fit ordinal logistic regression model
# model = LogisticIT()
# model.fit(X_train, y_train)
#
# # Predict
# y_pred = model.predict(X_test)
#
# # Evaluation
# print("Classification Report:\n")
# print(classification_report(y_test, y_pred, zero_division=0))
from library.core.statistical_model import StatisticalModel
import numpy as np

from sklearn import metrics
from sklearn.utils.validation import check_X_y
from library.under_development.stat_models.olr_helper._minimize import minimize
from library.utils.aggregation_client import AggregationClientInterface
from library.utils.numpy_aggregator import NumpyAggregator


class FedOrdinalLogisticRegression(StatisticalModel):
    """
            Classifier that implements the ordinal logistic model
            (Immediate-Threshold variant).

            Contrary to the OrdinalLogistic model, this variant
            minimizes a convex surrogate of the 0-1 loss, hence
            the score associated with this object is the accuracy
            score, i.e. the same score used in multiclass
            classification methods (sklearn.metrics.accuracy_score).

            Parameters
            ----------
            alpha: float
                Regularization parameter. Zero is no regularization, higher values
                increate the squared l2 regularization.

            References
            ----------
            J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
            Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
            Multidisciplinary Workshop on Advances in Preference Handling, 2005.
            """

    def __init__(self, client: AggregationClientInterface = None, alpha=1., verbose=False, max_iter=1000):
        super().__init__(client)
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter
        self.classes_ = None
        self.n_class_ = None
        self.coef_ = None
        self.theta_ = None
        self.agg = NumpyAggregator(self.client)

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(int)
        if self.agg.global_sum(np.abs(_y - y)) > 0.1:
            raise ValueError('y must only contain integer values')
        # T
        self.classes_ = np.unique(y)
        _min = self.agg.global_min(self.classes_).astype(int)
        _min = int(_min)
        self.n_class_ = self.agg.global_max(self.classes_) - _min + 1
        # Change to int
        self.n_class_ = np.array(self.n_class_, dtype=int)
        y_tmp = y - _min  # we need classes that start at zero
        self.coef_, self.theta_ = self.threshold_fit(
            X, y_tmp, self.alpha, self.n_class_,
            mode='0-1', verbose=self.verbose, max_iter=self.max_iter,
            sample_weight=sample_weight)
        return self

    def obj_margin(self, x0, X, y, alpha, n_class, weights, L, sample_weight):
        """
        Objective function for the general margin-based formulation
        """

        w = x0[:X.shape[1]]
        c = x0[X.shape[1]:]
        theta = L.dot(c)
        loss_fd = weights[y]
        _Xw = X.dot(w)
        _Alpha = theta[:, None] - _Xw  # (n_class - 1, n_samples)
        _S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

        err = loss_fd.T * log_loss(_S * _Alpha)
        if sample_weight is not None:
            err *= sample_weight
        obj = self.agg.fed_sum(np.sum(err))
        obj += alpha * 0.5 * (np.dot(w, w))
        return obj

    def grad_margin(self, x0, X, y, alpha, n_class, weights, L, sample_weight):
        """
        Gradient for the general margin-based formulation
        """

        w = x0[:X.shape[1]]
        c = x0[X.shape[1]:]
        theta = L.dot(c)
        loss_fd = weights[y]

        _Xw = X.dot(w)
        _Alpha = theta[:, None] - _Xw  # (n_class - 1, n_samples)
        _S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
        # Alpha[idx] *= -1
        # W[idx.T] *= -1

        _Sigma = _S * loss_fd.T * sigmoid(-_S * _Alpha)
        if sample_weight is not None:
            _Sigma *= sample_weight
        grad_w = self.agg.fed_sum(X.T.dot(_Sigma.sum(0))) + alpha * w

        grad_theta = self.agg.fed_sum(-_Sigma.sum(1))
        grad_c = L.T.dot(grad_theta)
        return np.concatenate((grad_w, grad_c), axis=0)

    def threshold_fit(self, x, y, alpha, n_class, mode='AE',
                      max_iter=1000, verbose=False, tol=1e-12,
                      sample_weight=None):
        # """
        # Solve the general threshold-based ordinal regression model
        # using the logistic loss as surrogate of the 0-1 loss
        #
        # Parameters
        # ----------
        # mode : string, one of {'AE', '0-1', 'SE'}
        #
        # """

        x, y = check_X_y(x, y, accept_sparse='csr')
        unique_y = np.sort(np.unique(y))
        if not np.all(unique_y == np.arange(unique_y.size)):
            raise ValueError(
                'Values in y must be %s, instead got %s'
                % (np.arange(unique_y.size), unique_y))

        n_samples, n_features = x.shape

        # convert from c to theta
        _L = np.zeros((n_class - 1, n_class - 1))
        _L[np.tril_indices(n_class - 1)] = 1.

        if mode == 'AE':
            # loss forward difference
            loss_fd = np.ones((n_class, n_class - 1))
        elif mode == '0-1':
            loss_fd = np.diag(np.ones(n_class - 1)) + \
                      np.diag(np.ones(n_class - 2), k=-1)
            loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
            loss_fd[-1, -1] = 1  # border case
        elif mode == 'SE':
            a = np.arange(n_class - 1)
            b = np.arange(n_class)
            loss_fd = np.abs((a - b[:, None]) ** 2 - (a - b[:, None] + 1) ** 2)
        else:
            raise NotImplementedError

        x0 = np.zeros(n_features + n_class - 1)
        x0[x.shape[1]:] = np.arange(n_class - 1)
        options = {'maxiter': max_iter, 'disp': verbose}
        if n_class > 2:
            bounds = [(None, None)] * (n_features + 1) + \
                     [(0, None)] * (n_class - 2)
        else:
            bounds = None

        sol = minimize(self.obj_margin, x0, method='L-BFGS-B',
                       jac=self.grad_margin, bounds=bounds, options=options,
                       args=(x, y, alpha, n_class, loss_fd, _L, sample_weight),
                       tol=tol)
        if verbose and not sol.success:
            print(sol.message)

        w, c = sol.x[:x.shape[1]], sol.x[x.shape[1]:]
        theta = _L.dot(c)
        return w, theta

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) + \
            self.classes_.min()

    def predict_proba(self, X):
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return metrics.accuracy_score(
            pred,
            y,
            sample_weight=sample_weight)


def log_loss(Z):
    # stable computation of the logistic loss
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))
    return out


def sigmoid(t):
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def threshold_predict(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    tmp = theta[:, None] - np.asarray(X.dot(w))
    pred = np.sum(tmp < 0, axis=0).astype(int)
    return pred


def threshold_proba(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1. Assumes
    the `sigmoid` link function is used.
    """
    eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)
    prob = np.pad(
        sigmoid(eta).T,
        pad_width=((0, 0), (1, 1)),
        mode='constant',
        constant_values=(0, 1))
    return np.diff(prob)
