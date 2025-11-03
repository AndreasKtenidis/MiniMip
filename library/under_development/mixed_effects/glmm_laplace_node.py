import numpy as np
from common_mixed_effect_functions import (
    pack_upper_triangle, clip_probs,
    logistic_sigmoid,
    glmm_binary_random_intercept_mode,
    glm_logistic_score_hessian_block,
    glmm_laplace_corrections_beta,
)

def glmm_binary_site_derivatives(
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    theta: np.ndarray,         
    w: np.ndarray | None = None,
    add_laplace_corrections: bool = True,
):
    if w is None:
        w = np.ones_like(y, dtype=float)

    n, p = X.shape
    q = int(theta.shape[0])
    beta = np.asarray(theta[:p], dtype=float)
    log_su2 = float(theta[p])
    sigma_u2 = float(np.exp(log_su2))
    inv_su2 = 1.0 / sigma_u2

    eta0 = X @ beta

    score = np.zeros(q, dtype=float)
    H = np.zeros((q, q), dtype=float)
    uniq = np.unique(center_ids)
    sum_logHuu = 0.0

    s_beta = score[:p]
    s_logsu2 = score[p:p+1]
    H_bb = H[:p, :p]
    H_bs = H[:p, p:p+1]
    H_ss = H[p:p+1, p:p+1]

    for j in uniq:
        idx = (center_ids == j)
        Xj = X[idx, :]
        yj = y[idx]
        wj = w[idx]
        eta_base = eta0[idx]

        # mode & Huu
        u_star, Huu = glmm_binary_random_intercept_mode(eta_base, yj, wj, sigma_u2)
        sum_logHuu += np.log(Huu)

        eta = eta_base + u_star
        pj = clip_probs(logistic_sigmoid(eta))

        # GLM block
        s_b, H_b = glm_logistic_score_hessian_block(Xj, yj, pj, wj)
        s_beta += s_b
        H_bb += H_b

        if add_laplace_corrections:
           
            corr_beta = glmm_laplace_corrections_beta(Xj, pj, wj, Huu)
            s_beta += corr_beta
           
            H_bs += 0.5 * (corr_beta[:, None]) * (inv_su2 / (Huu * Huu))

            
            s_logsu2[0] += 0.5 * ((u_star * u_star) * inv_su2 - 1.0) - 0.5 * (inv_su2 / Huu)
            
            H_ss[0, 0] += -0.5 * (u_star * u_star) * inv_su2 - 0.5 * (inv_su2 / Huu)

    H[p:p+1, :p] = H_bs.T

    return {
        "q": q,
        "score": score.astype(float).tolist(),
        "H_packed": pack_upper_triangle(H.astype(float)),
        "n_centers": int(len(uniq)),
        "sum_logHuu": float(sum_logHuu),
    }
