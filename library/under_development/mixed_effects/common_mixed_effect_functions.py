# common_mixed_effect_functions.py
from __future__ import annotations

from typing import Callable, Iterable, Tuple
import numpy as np


# ----------------------------- A) Validation & preprocessing ----------------------------- #

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    w: np.ndarray | None = None,
) -> None:
    """
    Basic validation for shapes, NaNs, finiteness, and alignment.

    Parameters
    ----------
    X : (n x p) float ndarray
    y : (n,)    float ndarray
    center_ids : (n,) 1D array (int-like or categorical codes)
    w : (n,) or None  float ndarray of non-negative weights

    Raises
    ------
    ValueError if any check fails.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    n, p = X.shape
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"y must be 1D with len=={n}, got shape {y.shape}.")
    if center_ids.ndim != 1 or center_ids.shape[0] != n:
        raise ValueError(f"center_ids must be 1D with len=={n}, got shape {center_ids.shape}.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf.")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN/Inf.")
    if w is not None:
        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError(f"w must be 1D with len=={n}, got shape {w.shape}.")
        if (w < 0).any() or not np.isfinite(w).all():
            raise ValueError("w must be non-negative and finite.")


def apply_weights(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply W^{1/2} premultiplication if weights are provided.

    Returns
    -------
    Xw : (n x p) ndarray
    yw : (n,)    ndarray
    """
    if w is None:
        return X, y
    s = np.sqrt(w).astype(float)
    return X * s[:, None], y * s


def extract_clusters(center_ids: np.ndarray) -> np.ndarray:
    """
    Sorted unique cluster labels present in this node.

    Returns
    -------
    clusters : 1D ndarray (unique, sorted)
    """
    return np.unique(center_ids)


def cluster_design_outcome(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    cluster_label,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice X, y for a given cluster.

    Returns
    -------
    Xj : (n_j x p) ndarray
    yj : (n_j,)    ndarray
    """
    idx = (center_ids == cluster_label)
    return X[idx], y[idx]


# ----------------------------- B) Random-intercept covariance utilities ----------------------------- #

def compute_vinv_random_intercept(
    nj: int,
    sigma2: float,
    sigma_u2: float,
) -> Tuple[np.ndarray, float]:
    """
    Closed-form inverse for V_j = sigma2*I + sigma_u2*11^T under random intercept.

    Returns
    -------
    Vj_inv : (n_j x n_j) ndarray
    alpha_j : float   (so that V_j^{-1} = (1/sigma2) I - alpha_j 11^T)
    """
    if nj <= 0:
        raise ValueError("nj must be positive.")
    if sigma2 <= 0 or sigma_u2 < 0:
        raise ValueError("Require sigma2>0 and sigma_u2>=0.")

    inv_sigma2 = 1.0 / sigma2
    alpha_j = sigma_u2 / (sigma2 * (sigma2 + nj * sigma_u2))
    ones = np.ones((nj, 1), dtype=float)
    Vj_inv = inv_sigma2 * np.eye(nj, dtype=float) - alpha_j * (ones @ ones.T)
    return Vj_inv, alpha_j


def logdet_v_random_intercept(nj: int, sigma2: float, sigma_u2: float) -> float:
    """
    log|V_j| for random-intercept covariance:
    log|V_j| = (n_j - 1) * log(sigma2) + log(sigma2 + n_j * sigma_u2)
    """
    if nj <= 0:
        raise ValueError("nj must be positive.")
    if sigma2 <= 0 or sigma_u2 < 0:
        raise ValueError("Require sigma2>0 and sigma_u2>=0.")
    return (nj - 1) * np.log(sigma2) + np.log(sigma2 + nj * sigma_u2)


# ----------------------------- C) GLS core summaries ----------------------------- #

def accumulate_gls_summaries(
    Sxx: np.ndarray,
    Sxy: np.ndarray,
    syy: float,
    Xj: np.ndarray,
    yj: np.ndarray,
    Vj_inv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    In-place accumulation of GLS summaries for a cluster j.

    Updates
    -------
    Sxx += Xj^T Vj_inv Xj     (p x p)
    Sxy += Xj^T Vj_inv yj     (p,)
    syy += yj^T Vj_inv yj     (scalar)
    """
    Sxx += Xj.T @ Vj_inv @ Xj
    Sxy += Xj.T @ Vj_inv @ yj
    syy += float(yj.T @ Vj_inv @ yj)
    return Sxx, Sxy, syy


# ----------------------------- D) REML gradient / score ingredients ----------------------------- #

def compute_cluster_residuals(
    Xj: np.ndarray,
    yj: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """
    r_j = y_j - X_j beta_hat
    """
    return yj - Xj @ beta_hat


def accumulate_reml_score_terms(
    q1: float,
    q2: float,
    T0: float,
    T1: float,
    B: np.ndarray,
    Xj: np.ndarray,
    Vj_inv: np.ndarray,
    rj: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    In-place accumulation of REML score ingredients for a cluster j.

    Updates
    -------
    q1 += ||Vj_inv rj||^2
    q2 += (1^T Vj_inv rj)^2
    T0 += tr(Vj_inv)
    T1 += 1^T Vj_inv 1
    B  += vj vj^T,  where vj = Xj^T Vj_inv 1_{n_j}
    """
    nj = Xj.shape[0]
    ones = np.ones((nj, 1), dtype=float)

    Vjr = Vj_inv @ rj
    q1 += float(Vjr @ Vjr)

    s1 = float(ones.T @ Vjr)
    q2 += s1 * s1

    T0 += float(np.trace(Vj_inv))
    T1 += float((ones.T @ Vj_inv @ ones).squeeze())

    vj = Xj.T @ (Vj_inv @ ones)   # (p x 1)
    B += vj @ vj.T                # (p x p, symmetric)
    return q1, q2, T0, T1, B


# ----------------------------- E) Histogram helpers for nj (optional, federated-friendly) ----------------------------- #

def build_local_hist(nj_sizes: Iterable[int], K: int | None = None) -> Tuple[int, np.ndarray]:
    """
    Build a local histogram of cluster sizes.

    Parameters
    ----------
    nj_sizes : iterable of positive ints (sizes per cluster in this node)
    K : optional int, if provided fixes the number of bins (1..K)

    Returns
    -------
    local_max : int           (max n_j in this node, 0 if none)
    hist : (K,) int ndarray   where hist[k-1] = count of clusters with size k
    """
    sizes = [int(x) for x in nj_sizes if int(x) > 0]
    if not sizes:
        return 0, np.zeros(0, dtype=np.int64)
    local_max = max(sizes)
    if K is None:
        K = local_max
    hist = np.zeros(K, dtype=np.int64)
    for n in sizes:
        if 1 <= n <= K:
            hist[n - 1] += 1
    return local_max, hist


def expand_hist_to_sizes(hist: np.ndarray) -> list[int]:
    """
    Convert a global histogram back to the explicit list of n_j's.

    Returns
    -------
    sizes : list[int] with multiplicities implied by the histogram.
    """
    sizes: list[int] = []
    for idx, cnt in enumerate(hist, start=1):
        if cnt > 0:
            sizes.extend([idx] * int(cnt))
    return sizes

#------------------------ F) utils of mixed effect ----------------------------------------#



def pack_upper_triangle(M: np.ndarray) -> list[float]:
    """
    Return the upper triangle (including diagonal) of a symmetric matrix M as a flat list[float].
    Length = p*(p+1)//2 for M in R^{p x p}. Protobuf-friendly for repeated double.
    """
    q = M.shape[0]
    iu = np.triu_indices(q)
    return M[iu].astype(np.float64).tolist()

def unpack_upper_triangle(v: list[float], p: int) -> np.ndarray:
    """
    Reconstruct a symmetric p x p matrix from a flat list[float] that stores the upper triangle.
    """
    arr = np.asarray(v, dtype=np.float64)
    M = np.zeros((p, p), dtype=np.float64)
    iu = np.triu_indices(p)
    M[iu] = arr
    M[(iu[1], iu[0])] = arr  
    return M

def clip_probs(p, eps: float = 1e-8):
    """
    Clip probabilities to (eps, 1-eps). Useful in GLM/GLMM with logits to avoid log(0).
    """
    return np.clip(p, eps, 1.0 - eps)

def ridge_invert(A: np.ndarray, lam: float = 1e-8) -> np.ndarray:
    """
    Return a Tikhonov-regularized inverse (pseudo-inverse) of A:
    (A + lam I)^{+}. Stable when A is near-singular.
    """
    return np.linalg.pinv(A + lam * np.eye(A.shape[0], dtype=A.dtype))


# ------------------------------- GLS / linear algebra ------------------------------- #

def gls_beta_from_Sxx_Sxy(
    Sxx: np.ndarray,
    Sxy: np.ndarray,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve (Sxx) beta = Sxy with Tikhonov ridge for stability.

    Returns
    -------
    beta : (p,) ndarray
    Ainv : (p x p) ndarray, inverse (pseudo-inverse) of Sxx with ridge
    """
    p = Sxx.shape[0]
    A = Sxx + ridge * np.eye(p, dtype=Sxx.dtype)
    Ainv = np.linalg.pinv(A)
    beta = Ainv @ Sxy
    return beta, Ainv


# -------------------------- log|V| for random-intercept structure ------------------- #

def sum_logdet_random_intercept_from_sizes(
    nj_sizes: Iterable[int],
    sigma2: float,
    sigma_u2: float,
) -> float:
    """
    Compute sum_j log|V_j| with V_j = sigma2*I + sigma_u2*11^T for given {n_j}.

    log|V_j| = (n_j - 1) * log(sigma2) + log(sigma2 + n_j * sigma_u2)
    """
    if sigma2 <= 0 or sigma_u2 < 0:
        raise ValueError("Require sigma2>0 and sigma_u2>=0.")
    s = 0.0
    for nj in nj_sizes:
        nj = int(nj)
        if nj <= 0:
            continue
        s += (nj - 1) * np.log(sigma2) + np.log(sigma2 + nj * sigma_u2)
    return float(s)


def sum_logdet_random_intercept_from_hist(
    hist: np.ndarray,
    sigma2: float,
    sigma_u2: float,
) -> float:
    """
    Same as above, but using a histogram over n=1..K:
    hist[k-1] = number of clusters with size k.
    """
    if sigma2 <= 0 or sigma_u2 < 0:
        raise ValueError("Require sigma2>0 and sigma_u2>=0.")
    if hist.size == 0:
        return 0.0
    k = np.arange(1, hist.size + 1, dtype=float)  # 1..K
    terms = (k - 1.0) * np.log(sigma2) + np.log(sigma2 + k * sigma_u2)
    return float(np.dot(hist.astype(float), terms))


# ------------------------------ REML objective / gradient --------------------------- #

def reml_objective_from_summaries(
    Sxx: np.ndarray,
    syy: float,
    beta: np.ndarray,
    *,
    sigma2: float,
    sigma_u2: float,
    nj_sizes: Iterable[int] | None = None,
    hist: np.ndarray | None = None,
) -> float:
    """
    ℓ_REML(σ^2, σ_u^2) = -1/2 [ sum log|V_j| + log|Sxx| + syy - beta^T Sxx beta ].
    Provide either nj_sizes or hist 
    """
    if (nj_sizes is None) and (hist is None):
        raise ValueError("Provide either nj_sizes or hist.")
    if (nj_sizes is not None) and (hist is not None):
        raise ValueError("Provide only one of nj_sizes or hist, not both.")

    # sum log|V|
    if nj_sizes is not None:
        logdet_V = sum_logdet_random_intercept_from_sizes(nj_sizes, sigma2, sigma_u2)
    else:
        logdet_V = sum_logdet_random_intercept_from_hist(hist, sigma2, sigma_u2)

    # log|Sxx|
    
    sign, logdet_Sxx = np.linalg.slogdet(Sxx + 1e-12 * np.eye(Sxx.shape[0], dtype=Sxx.dtype))
    if sign <= 0:
      
        logdet_Sxx = np.log(np.linalg.det(Sxx + 1e-6 * np.eye(Sxx.shape[0], dtype=Sxx.dtype)))

    quad = float(beta @ (Sxx @ beta))
    ell = -0.5 * (logdet_V + logdet_Sxx + float(syy) - quad)
    return float(ell)


def reml_grad_logscale_from_summaries(
    q1: float,
    q2: float,
    T0: float,
    T1: float,
    B: np.ndarray,
    Ainv: np.ndarray,
    p: int,
    sigma2: float,
    sigma_u2: float,
) -> np.ndarray:
    """
    Gradient wrt (log sigma2, log sigma_u2).

    U_{sigma2}  = 0.5 * [ q1 - (T0 - p) ]
    U_{sigma_u2}= 0.5 * [ q2 - ( T1 - tr(Ainv B) ) ]
    g = [ sigma2 * U_{sigma2},  sigma_u2 * U_{sigma_u2} ]
    """
    tr_Ainv_B = float(np.trace(Ainv @ B))
    U_s2 = 0.5 * (float(q1) - (float(T0) - float(p)))
    U_su2 = 0.5 * (float(q2) - (float(T1) - tr_Ainv_B))
    g1 = float(sigma2) * U_s2
    g2 = float(sigma_u2) * U_su2
    return np.array([g1, g2], dtype=np.float64)


# --------------------------- generic 2D log-scale line-search ----------------------- #

def backtracking_line_search_log2d(
    ell_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    *,
    alpha: float = 1.0,
    c: float = 1e-4,
    max_backtracks: int = 8,
    min_step: float = 1e-6,
    lower_bound: float = 1e-12,
) -> Tuple[np.ndarray, bool]:
    """
    Generic backtracking line-search in 2D on log-parameters.
    phi = [log sigma2, log sigma_u2].

    Returns
    -------
    phi_new : (2,) ndarray
    improved : bool
    """
    phi0 = np.asarray(phi0, dtype=float).reshape(2)
    ell0 = float(ell_fn(phi0))
    g = np.asarray(grad_fn(phi0), dtype=float).reshape(2)
    d = -g  

    if np.allclose(g, 0.0):
        return phi0, True

    step = float(alpha)
    for _ in range(max_backtracks):
        phi = phi0 + step * d
        s2, su2 = float(np.exp(phi[0])), float(np.exp(phi[1]))
        if (s2 < lower_bound) or (su2 < lower_bound):
            step *= 0.5
            if step < min_step:
                return phi0, False
            continue

        ell = float(ell_fn(phi))
        if ell >= ell0 + c * step * float(g @ d):  
            return phi, True

        step *= 0.5
        if step < min_step:
            break

    return phi0, False


####################### GLMM helpers ##################################



def logistic_sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def glmm_binary_random_intercept_mode(
    eta_base: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    sigma_u2: float,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """
    Newton mode for u_j in logistic GLMM with random intercept.
    Returns (u_star, Huu) with Huu = sum w p(1-p) + 1/sigma_u2 at the mode.
    """
    if w is None:
        w = np.ones_like(y, dtype=float)
    inv_su2 = 1.0 / sigma_u2
    u = 0.0
    for _ in range(max_iter):
        eta = eta_base + u
        p = clip_probs(logistic_sigmoid(eta))
        g = np.sum(w * (y - p)) - u * inv_su2
        h = -np.sum(w * p * (1.0 - p)) - inv_su2
        step = g / h
        u_new = u - step
        if abs(u_new - u) < tol:
            u = u_new
            break
        u = u_new
    Huu = -h
    return float(u), float(Huu)

def glm_logistic_score_hessian_block(
    Xj: np.ndarray, yj: np.ndarray, pj: np.ndarray, wj: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score/Hessian block for logistic regression on a subset j with weights wj.
    Returns (s_beta, H_bb).
    """
    rj = (yj - pj) * wj               
    Vj = (pj * (1.0 - pj)) * wj       
    s_beta = Xj.T @ rj
    H_bb = -(Xj.T * Vj) @ Xj
    return s_beta, H_bb

def glmm_laplace_corrections_beta(
    Xj: np.ndarray, pj: np.ndarray, wj: np.ndarray, Huu: float
) -> np.ndarray:
    """
    0.5 * Huu^{-1} * sum_i w p(1-p)(1-2p) x_i   (correction to score wrt beta).
    """
    dHuu_deta = (pj * (1.0 - pj)) * (1.0 - 2.0 * pj) * wj
    return 0.5 * (Xj.T @ dHuu_deta) / float(Huu)


