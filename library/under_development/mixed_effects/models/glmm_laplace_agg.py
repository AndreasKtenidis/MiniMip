# glmm_laplace_agg.py
import numpy as np
from .utils import unpack_upper_triangle, ridge_invert

class GLMMAggregator:
    def __init__(self, q):
        self.q = q
        self.reset()

    def reset(self):
        self.s = np.zeros(self.q)
        self.H = np.zeros((self.q,self.q))

    def accumulate(self, msg):
        self.s += np.array(msg.score)
        self.H += unpack_upper_triangle(msg.H_packed, self.q)

    def newton_update(self, theta, ridge=1e-6):
        H_r = self.H + ridge*np.eye(self.q)
        delta = np.linalg.solve(H_r, self.s)
        theta_new = theta - delta
        return theta_new

    def converged(self, theta, theta_new, s, tol_theta=1e-6, tol_score=1e-6):
        return (np.max(np.abs(theta_new-theta)) < tol_theta) and (np.linalg.norm(s) < tol_score)
