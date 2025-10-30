# iptw.py
import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_propensity(X, T):
    mdl = LogisticRegression(max_iter=1000, solver="lbfgs")
    mdl.fit(X, T)
    e = mdl.predict_proba(X)[:,1]
    return e

def stabilized_ate_weights(T, e, trim=None, cap=None):
    p1 = T.mean(); p0 = 1.0 - p1
    w = np.where(T==1, p1/np.maximum(e,1e-8), p0/np.maximum(1-e,1e-8))
    if trim:
        lo, hi = np.quantile(w, [trim, 1-trim])
        keep = (w>=lo) & (w<=hi)
        w = w[keep]
    if cap:
        w = np.clip(w, 0, cap)
    return w
