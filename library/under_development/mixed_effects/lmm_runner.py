# mini_mip_system/client/lmm_reml_runner.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from mini_mip_system.client.grpc_agg_client import GRPCClient
from lmm_reml_node import reml_summaries_node
from lmm_reml_aggregator import LMMRemlAggregator
from utils.numpy_aggregator import NumpyAggregator  


from library.under_development.mixed_effects.common_mixed_effect_functions import (
    build_local_hist,
    expand_hist_to_sizes,
)
def run_lmm_reml_round(
    client: GRPCClient,
    *,
    operation_id: int,      
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    beta_hat: np.ndarray,   
    sigma2: float,
    sigma_u2: float,
    w: Optional[np.ndarray] = None,
    ridge: float = 1e-8,
    lower_bound: float = 1e-6,   
) -> Dict[str, Any]:
    """
    One federated LMM/REML round in TWO phases:

    Phase A (GLS):
      Node: compute GLS summaries (Sxx, Sxy, syy) + nj_sizes
      Agg : federated SUMs -> accumulate -> compute beta_GLS, Ainv

    Phase B (REML score):
      Broadcast beta_GLS (conceptually; here we reuse local X,y with beta_GLS)
      Node: compute ONLY REML score terms (q1, q2, T0, T1, B) using beta_GLS
      Agg : federated SUMs on q-terms -> update (sigma2, sigma_u2) with line-search

    Then do θ-consensus (beta_GLS, sigma2_new, sigma_u2_new) via fed_sum to all nodes.

    Returns
    -------
    Dict with keys:
      - beta, sigma2, sigma_u2, improved, global_hist, aggregates{Sxx_packed,Sxy,syy,q,B_packed}
    """
    npagg = NumpyAggregator(client)

    # ----------------------------- Phase A: GLS summaries ----------------------------- 
    # Use node-side routine once to get Sxx/Sxy/syy and cluster sizes (ignore q-terms here).
    node_payload_A = reml_summaries_node(
        X=X, y=y, center_ids=center_ids,
        sigma2=sigma2, sigma_u2=sigma_u2,
        beta_hat=(beta_hat if beta_hat is not None else np.zeros(X.shape[1], dtype=float)),
        w=w
    )

    # Decide histogram length via global MAX of local_max
    local_max, _ = build_local_hist(node_payload_A.get("nj_sizes", []))
    global_max = int(npagg.global_max(np.array([local_max], dtype=np.int64))[0])

    # Federated SUM of histograms (length = global_max)
    if global_max > 0:
        _, local_hist = build_local_hist(node_payload_A.get("nj_sizes", []), K=global_max)
        global_hist = npagg.fed_sum(local_hist.astype(np.float64)).astype(np.int64)
    else:
        global_hist = np.zeros(0, dtype=np.int64)

    # Federated SUMs for GLS pieces (packed Sxx, Sxy, syy)
    Sxx_sum = npagg.fed_sum(np.asarray(node_payload_A["Sxx_packed"], dtype=np.float64))
    Sxy_sum = npagg.fed_sum(np.asarray(node_payload_A["Sxy"],        dtype=np.float64))
    syy_sum = npagg.fed_sum(np.asarray([node_payload_A["syy"]],      dtype=np.float64))

    # Aggregator: build state with GLS-only info and compute beta_GLS
    p = int(node_payload_A["p"])
    agg = LMMRemlAggregator(p)
    agg.reset()
    agg.accumulate({
        "Sxx_packed": Sxx_sum,
        "Sxy":        Sxy_sum,
        "syy":        float(syy_sum[0]),
        "global_hist": global_hist,   
       
    })
    beta_gls, Ainv = agg.compute_beta_gls(ridge=ridge)

    # ----------------------- Phase B: REML score terms with beta_GLS ------------------ #
    # Recompute ONLY q-terms / B using the CURRENT beta_gls (same sigma2,sigma_u2 of this round)
    node_payload_B = reml_summaries_node(
        X=X, y=y, center_ids=center_ids,
        sigma2=sigma2, sigma_u2=sigma_u2,
        beta_hat=beta_gls,   #  residuals r_j = y_j - X_j beta_gls
        w=w
    )

    # Federated SUMs for q-terms / B only (ignore Sxx/Sxy/syy here to avoid double counting)
    q_sum = npagg.fed_sum(np.asarray([
        node_payload_B["q1_norm_vinv_r_sq"],
        node_payload_B["q2_1t_vinv_r_sq"],
        node_payload_B["T0_tr_vinv_sum"],
        node_payload_B["T1_1t_vinv_1"]
    ], dtype=np.float64))
    B_sum = npagg.fed_sum(np.asarray(node_payload_B["B_v_outer_sum_packed"], dtype=np.float64))

    # Accumulate q-terms into the same aggregator (Sxx/Sxy/syy are already there from Phase A)
    agg.accumulate({
        "q1_norm_vinv_r_sq": float(q_sum[0]),
        "q2_1t_vinv_r_sq":   float(q_sum[1]),
        "T0_tr_vinv_sum":    float(q_sum[2]),
        "T1_1t_vinv_1":      float(q_sum[3]),
        "B_v_outer_sum_packed": B_sum,
        
    })

    # ---------------------------- Variance update (REML) ------------------------------ 
    sigma2_new, sigma_u2_new, improved = agg.update_variances(
        beta=beta_gls, Ainv=Ainv, sigma2=sigma2, sigma_u2=sigma_u2,
        lower_bound=lower_bound
    )

    # ----------------------------- Consensus (θ broadcast) ---------------------------- 
    theta_vec = np.concatenate(
        [np.asarray(beta_gls, dtype=np.float64),
         np.array([sigma2_new, sigma_u2_new], dtype=np.float64)]
    )
    is_coord = getattr(client, "client_id", 0) == 0
    theta_send = theta_vec if is_coord else np.zeros_like(theta_vec)
    theta_global = npagg.fed_sum(theta_send)  # all nodes receive the same θ

    beta_cons     = theta_global[:p]
    sigma2_cons   = float(theta_global[p])
    sigma_u2_cons = float(theta_global[p + 1])

    return {
        "beta": beta_cons,
        "sigma2": sigma2_cons,
        "sigma_u2": sigma_u2_cons,
        "improved": bool(improved),
        "global_hist": global_hist,
        "aggregates": {
            "Sxx_packed": np.asarray(Sxx_sum, dtype=np.float64),
            "Sxy":        np.asarray(Sxy_sum, dtype=np.float64),
            "syy":        float(syy_sum[0]),
            "q":          np.asarray(q_sum, dtype=np.float64),
            "B_packed":   np.asarray(B_sum, dtype=np.float64),
        },
    }
