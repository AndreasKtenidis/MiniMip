# glmm_laplace_agg.py
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple, Optional

from common_mixed_effect_functions import unpack_upper_triangle

class GLMMAggregator:
    """
    Aggregator for GLMM (binary, Laplace) Newton steps on theta = [beta..., log_sigma_u2].

    Typical outer iteration:
      agg.reset()
      for each site:
          payload = { "score": list/ndarray length q, "H_packed": upper-tri list/ndarray }
          agg.accumulate(payload)
      theta_new = agg.newton_update(theta, ridge=1e-6, ...)
      if agg.converged(theta, theta_new, agg.s, ...): stop
      # broadcast theta_new and repeat
    """

    def __init__(self, q: int):
        self.q = int(q)
        self.reset()

    # --------------------------- state --------------------------- 
    def reset(self) -> None:
        self.s: np.ndarray = np.zeros(self.q, dtype=np.float64)                 # global score
        self.H: np.ndarray = np.zeros((self.q, self.q), dtype=np.float64)       # global Hessian (approx)
        self._n_msgs: int = 0

    # ------------------------ accumulate ------------------------ 
    def accumulate(self, payload: Dict[str, Any]) -> None:
        """
        Accumulate one node's derivatives.

        Parameters
        ----------
        payload: dict
            Must contain:
              - "score": array-like, shape (q,)
              - "H_packed": array-like, packed upper triangle of shape (q*(q+1)//2,)
        """
        if not isinstance(payload, dict):
            raise TypeError("accumulate expects a dict payload")

        score = payload.get("score", None)
        H_packed = payload.get("H_packed", None)
        if score is None or H_packed is None:
            raise ValueError("payload must include 'score' and 'H_packed' keys")

        s = np.asarray(score, dtype=np.float64).reshape(-1)
        if s.shape[0] != self.q:
            raise ValueError(f"score length {s.shape[0]} != q={self.q}")

        H_block = unpack_upper_triangle(list(H_packed), self.q)
        # Symmetrize for numerical stability
        H_block = 0.5 * (H_block + H_block.T)

        self.s += s
        self.H += H_block
        self._n_msgs += 1

    # ----------------------- Newton update ---------------------- #
    def newton_update(
        self,
        theta: np.ndarray,
        ridge: float = 1e-6,
        *,
        max_tries: int = 6,
        max_step_norm: float = 5.0,
        clip_log_sigma_bounds: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Compute theta_new = theta - (H + λ I)^{-1} s with Levenberg–Marquardt damping.

        Parameters
        ----------
        theta : ndarray, shape (q,)
        ridge : float
            Initial λ for damping. Will be increased if the linear solve fails.
        max_tries : int
            Max attempts with increasing λ.
        max_step_norm : float
            Euclidean-norm cap for the step to avoid huge jumps.
        clip_log_sigma_bounds : (low, high) or None
            If provided, clip the last parameter (log σ_u^2) into [low, high].

        Returns
        -------
        theta_new : ndarray, shape (q,)
        """
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.shape[0] != self.q:
            raise ValueError(f"theta length {theta.shape[0]} != q={self.q}")

        # Symmetrize global H and copy score
        H = 0.5 * (self.H + self.H.T)
        s = self.s.copy()

        I = np.eye(self.q, dtype=np.float64)
        lam = float(max(ridge, 0.0))
        delta = None

        for _ in range(int(max_tries)):
            try:
                H_reg = H + lam * I
                delta = np.linalg.solve(H_reg, s)     # solve H_reg * delta = s
                break
            except np.linalg.LinAlgError:
                lam = max(lam * 10.0, 1e-12)         # bump ridge and retry

        if delta is None:
           
            H_reg = H + max(lam, 1e-6) * I
            delta = np.linalg.pinv(H_reg) @ s

        # Step-size clipping
        step_norm = float(np.linalg.norm(delta))
        if step_norm > max_step_norm and step_norm > 0.0:
            delta = delta * (max_step_norm / step_norm)

        theta_new = theta - delta

    
        if clip_log_sigma_bounds is not None:
            lo, hi = clip_log_sigma_bounds
            theta_new[-1] = float(np.clip(theta_new[-1], lo, hi))

        return theta_new

    # ---------------------- convergence check ------------------- #
    @staticmethod
    def converged(
        theta: np.ndarray,
        theta_new: np.ndarray,
        score: np.ndarray,
        *,
        tol_theta: float = 1e-6,
        tol_score: float = 1e-6,
    ) -> bool:
        """
        Converged if both:
          max_i |theta_new[i] - theta[i]| < tol_theta
          and   ||score||_2 < tol_score
        """
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        theta_new = np.asarray(theta_new, dtype=np.float64).reshape(-1)
        score = np.asarray(score, dtype=np.float64).reshape(-1)

        dtheta_max = float(np.max(np.abs(theta_new - theta)))
        snorm = float(np.linalg.norm(score))
        return (dtheta_max < tol_theta) and (snorm < tol_score)
