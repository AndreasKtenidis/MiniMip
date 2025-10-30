# mini_mip_system/models/lmm_reml_node.py
from __future__ import annotations
import numpy as np

from common_mixed_effect_functions import pack_upper_triangle  
from common_mixed_effect_functions import (
    validate_inputs,
    apply_weights,
    extract_clusters,
    cluster_design_outcome,
    compute_vinv_random_intercept,
    accumulate_gls_summaries,
    compute_cluster_residuals,
    accumulate_reml_score_terms,
    logdet_v_random_intercept,
)

def reml_summaries_node(X, y, center_ids, sigma2, sigma_u2, beta_hat, w=None):
    """Site-level summaries for REML (random-intercept LMM), using common helpers."""
    
    validate_inputs(X, y, center_ids, w)
    Xw, yw = apply_weights(X, y, w)

    p = X.shape[1]
    Sxx = np.zeros((p, p), dtype=np.float64)
    Sxy = np.zeros(p, dtype=np.float64)
    syy = 0.0
    logdet_sum = 0.0

    q1 = 0.0
    q2 = 0.0
    T0 = 0.0
    T1 = 0.0
    B  = np.zeros((p, p), dtype=np.float64)

    nj_sizes: list[int] = []
    clusters = extract_clusters(center_ids)

    for j in clusters:
        Xj, yj = cluster_design_outcome(Xw, yw, center_ids, j)
        nj = int(Xj.shape[0])
        if nj == 0:
            continue
        nj_sizes.append(nj)

        Vj_inv, _ = compute_vinv_random_intercept(nj, sigma2, sigma_u2)

        # GLS
        Sxx, Sxy, syy = accumulate_gls_summaries(Sxx, Sxy, syy, Xj, yj, Vj_inv)
        logdet_sum += logdet_v_random_intercept(nj, sigma2, sigma_u2)

        # REML score terms
        rj = compute_cluster_residuals(Xj, yj, beta_hat)
        q1, q2, T0, T1, B = accumulate_reml_score_terms(
            q1, q2, T0, T1, B, Xj, Vj_inv, rj
        )

    payload = {
        "p": p,
        "Sxx_packed": pack_upper_triangle(Sxx),
        "Sxy": Sxy.tolist(),
        "syy": float(syy),
        "logdet_sum": float(logdet_sum),
        "q1_norm_vinv_r_sq": float(q1),
        "q2_1t_vinv_r_sq": float(q2),
        "T0_tr_vinv_sum": float(T0),
        "T1_1t_vinv_1": float(T1),
        "B_v_outer_sum_packed": pack_upper_triangle(B),
        "nj_sizes": nj_sizes,
    }
    return payload
