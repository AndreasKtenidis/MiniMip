# glmm_laplace_node.py
import numpy as np
from .utils import pack_upper_triangle, clip_probs

def _binary_mode_center(eta_base, y, w, sigma_u2, max_iter=25, tol=1e-8):
    # eta = eta_base + u
    u = 0.0
    for _ in range(max_iter):
        eta = eta_base + u
        p = clip_probs(1.0/(1.0 + np.exp(-eta)))
        g = np.sum(w*(y - p)) - u/sigma_u2
        h = -np.sum(w*p*(1-p)) - 1.0/sigma_u2
        step = g / h
        u_new = u - step
        if abs(u_new - u) < tol: 
            u = u_new; break
        u = u_new
    Huu = -h
    return u, Huu

def glmm_binary_site_derivatives(X, y, T, center_ids, theta, w):
    # theta = [beta, log_sigma_u2]
    q = len(theta)
    beta = np.array(theta[:-1]); sigma_u2 = np.exp(theta[-1])
    score = np.zeros(q); H = np.zeros((q,q))

    # precompute linear predictor baseline (without u)
    eta0 = X @ beta

    # per center: mode u*_j
    modes = {}
    for j in np.unique(center_ids):
        idx = (center_ids==j)
        u_star, Huu = _binary_mode_center(eta0[idx], y[idx], w[idx], sigma_u2)
        modes[j] = (u_star, Huu)

        # contribute to score/Hessian wrt beta
        eta = eta0[idx] + u_star
        p = clip_probs(1.0/(1.0 + np.exp(-eta)))
        r = (y[idx] - p) * w[idx]
        V = (p*(1-p)) * w[idx]
        Xj = X[idx]
        s_beta = Xj.T @ r
        H_bb = -(Xj.T * V) @ Xj
        score[:-1] += s_beta
        H[:-1,:-1] += H_bb

    # variance component 
    u_vec = np.array([modes[j][0] for j in modes])
    score[-1] = 0.5*(np.sum(u_vec**2)/sigma_u2 - len(modes))
    H[-1,-1] = -0.5*len(modes)  # approx on log-scale
   

    return {
        "q": q,
        "score": score.tolist(),
        "H_packed": pack_upper_triangle(H)
    }
