"""Convergence helpers for Monte Carlo retirement studies (Optuna, etc.)."""

from __future__ import annotations

import math
from typing import Final

import numpy as np

# Defaults aligned with tuning scripts; import these constants in objectives.
MIN_SIMS_FOR_CONVERGENCE: Final[int] = 200_000
PROB_SUCCESS_SE_ACCEPTANCE: Final[float] = 0.001
PROB_SUCCESS_SE_DIFF_THRESHOLD: Final[float] = 0.01
WILSON_95_Z: Final[float] = 1.96


def wilson_95_half_width(wilson_95: tuple[float, float]) -> float:
    """Half-width of a Wilson interval in probability units [0, 1]."""
    lo, hi = wilson_95
    return (hi - lo) / 2.0


def assess_wilson_95_precision(
    wilson_95: tuple[float, float],
    *,
    se_acceptance: float = PROB_SUCCESS_SE_ACCEPTANCE,
    z: float = WILSON_95_Z,
) -> tuple[bool, float, float]:
    """
    Return (is_ok, half_width, target_half_width) for P(success).

    OK when the Wilson 95% half-width is at most z * se_acceptance — the same
    precision target used by Optuna convergence stopping (SE <= 0.001).
    """
    half_width = wilson_95_half_width(wilson_95)
    target_half_width = z * se_acceptance
    return (half_width <= target_half_width, half_width, target_half_width)


def cumulative_success_metrics(
    final_balances: np.ndarray,
) -> tuple[float, float, int]:
    """
    Return (p_hat, standard_error_of_p, success_count) for Bernoulli(success).

    Uses sqrt(p*(1-p)/n). For n==0 returns (nan, nan, 0).
    """
    n = int(final_balances.size)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    successes = int(np.sum(final_balances > 0))
    p_hat = successes / n
    se = math.sqrt(max(p_hat * (1.0 - p_hat) / n, 0.0))
    return (float(p_hat), float(se), successes)


def prob_convergence_should_stop(
    *,
    n_done: int,
    prob_se: float,
    prev_prob_se: float,
    min_sims: int = MIN_SIMS_FOR_CONVERGENCE,
    se_acceptance: float = PROB_SUCCESS_SE_ACCEPTANCE,
    se_diff_threshold: float = PROB_SUCCESS_SE_DIFF_THRESHOLD,
) -> tuple[bool, str]:
    """
    Stop when enough sims and either success-rate SE is small enough,
    or relative improvement in SE is below threshold (vs previous batch).
    """
    if n_done < min_sims:
        return (False, "below_min_sims")
    if not math.isfinite(prob_se):
        return (False, "non_finite_se")
    if prob_se <= se_acceptance:
        return (True, "prob_se_acceptance")
    if prev_prob_se != float("inf") and prev_prob_se > 0.0:
        rel_improve = (prev_prob_se - prob_se) / prev_prob_se
        if rel_improve <= se_diff_threshold:
            return (True, "prob_se_stabilized")
    return (False, "continue")
