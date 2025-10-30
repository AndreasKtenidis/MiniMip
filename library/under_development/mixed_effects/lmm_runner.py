# mini_mip_system/client/lmm_reml_runner.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from mini_mip_system.client.grpc_agg_client import GRPCClient
from lmm_reml_node import reml_summaries_node
from lmm_reml_aggregator import LMMRemlAggregator
from library.utils.numpy_aggregator import NumpyAggregator  


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
) -> Dict[str, Any]:
    """
    One federated LMM/REML round:
      Node: compute summaries → int global_max for K → fed_sum hist → fed_sum floats
      Aggregator: accumulate dict → GLS → REML line-search
      Consensus: coordinator broadcasts θ via float-SUM
    """
    
    npagg = NumpyAggregator(client)

    # 1) Node-side summaries
    node_payload = reml_summaries_node(
        X=X, y=y, center_ids=center_ids,
        sigma2=sigma2, sigma_u2=sigma_u2,
        beta_hat=beta_hat, w=w
    )

    # 2) int MAX to decide histogram bins (K)
    local_max, _ = build_local_hist(node_payload.get("nj_sizes", []))
    
    global_max = int(npagg.global_max(np.array([local_max], dtype=np.int64))[0])

    # 3) SUM histograms , length=global_max
    if global_max > 0:
        _, local_hist = build_local_hist(node_payload.get("nj_sizes", []), K=global_max)
        
        global_hist = npagg.fed_sum(local_hist.astype(np.float64)).astype(np.int64)
    else:
        global_hist = np.zeros(0, dtype=np.int64)

    # 4) Federated float SUMs for packed Sxx, Sxy, syy, q's, B_packed
    Sxx_sum = npagg.fed_sum(np.asarray(node_payload["Sxx_packed"], dtype=np.float64))
    Sxy_sum = npagg.fed_sum(np.asarray(node_payload["Sxy"],        dtype=np.float64))
    syy_sum = npagg.fed_sum(np.asarray([node_payload["syy"]],      dtype=np.float64))
    q_sum   = npagg.fed_sum(np.asarray([
        node_payload["q1_norm_vinv_r_sq"],
        node_payload["q2_1t_vinv_r_sq"],
        node_payload["T0_tr_vinv_sum"],
        node_payload["T1_1t_vinv_1"]
    ], dtype=np.float64))
    B_sum   = npagg.fed_sum(np.asarray(node_payload["B_v_outer_sum_packed"], dtype=np.float64))

    # 5) Aggregator step 
    p = int(node_payload["p"])
    agg = LMMRemlAggregator(p)
    agg.reset()

    agg_payload: Dict[str, Any] = {
        "Sxx_packed": Sxx_sum,
        "Sxy":        Sxy_sum,
        "syy":        float(syy_sum[0]),
        "q1_norm_vinv_r_sq": float(q_sum[0]),
        "q2_1t_vinv_r_sq":   float(q_sum[1]),
        "T0_tr_vinv_sum":    float(q_sum[2]),
        "T1_1t_vinv_1":      float(q_sum[3]),
        "B_v_outer_sum_packed": B_sum,
        "global_hist": global_hist,   
        # "nj_sizes": []              
    }
    agg.accumulate(agg_payload)

    beta_gls, Ainv = agg.compute_beta_gls(ridge=ridge)

    sigma2_new, sigma_u2_new, improved = agg.update_variances(
        beta=beta_gls, Ainv=Ainv, sigma2=sigma2, sigma_u2=sigma_u2
    )

    # 6) θ-consensus via float-SUM - coordinator sends, others send zeros 
    theta_vec = np.concatenate(
        [np.asarray(beta_gls, dtype=np.float64),
         np.array([sigma2_new, sigma_u2_new], dtype=np.float64)]
    )
    is_coord = getattr(client, "client_id", 0) == 0
    theta_send = theta_vec if is_coord else np.zeros_like(theta_vec)
    
    # sum across nodes and everyone gets the same θ
    theta_global = npagg.fed_sum(theta_send)  

    beta_cons   = theta_global[:p]
    sigma2_cons = float(theta_global[p])
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
