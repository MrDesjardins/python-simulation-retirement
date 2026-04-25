"""Tests for reliability metrics, convergence helpers, and scenario wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common import (
    SimulationData,
    binomial_proportion_standard_error,
    run_simulation_mp,
    wilson_score_interval,
)
from simulation_convergence import (
    PROB_SUCCESS_SE_ACCEPTANCE,
    cumulative_success_metrics,
    prob_convergence_should_stop,
)


def test_wilson_score_interval_bounds() -> None:
    lo, hi = wilson_score_interval(50, 100, z=1.96)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 0.55 < hi


def test_binomial_se_at_extremes() -> None:
    assert binomial_proportion_standard_error(0.5, 10000) == pytest.approx(0.005, rel=1e-6)
    se0 = binomial_proportion_standard_error(0.0, 1000)
    assert se0 == 0.0
    se1 = binomial_proportion_standard_error(1.0, 1000)
    assert se1 == 0.0


def test_cumulative_success_metrics() -> None:
    arr = np.array([1.0, -1.0, 10.0, 0.0])
    p, se, k = cumulative_success_metrics(arr)
    assert k == 2
    assert p == 0.5
    assert se == pytest.approx(np.sqrt(0.5 * 0.5 / 4))


def test_prob_convergence_should_stop() -> None:
    ok, _ = prob_convergence_should_stop(
        n_done=200_000,
        prob_se=0.0005,
        prev_prob_se=float("inf"),
        se_acceptance=PROB_SUCCESS_SE_ACCEPTANCE,
    )
    assert ok is True


def test_simulation_data_reserve_floor() -> None:
    fb = np.array([1_000_000.0, 0.0, 400_000.0])
    r = np.array([0.01, 0.02])
    sd = SimulationData(
        1_000_000,
        50_000,
        50_000,
        30,
        3,
        fb,
        None,
        100,
        r,
        reserve_floor=500_000,
    )
    # 0 and 400k are strictly below 500k floor
    assert sd.probability_below_reserve_floor == pytest.approx(2.0 / 3.0)
    assert sd.success_count == 2
    assert sd.probability_of_success_se > 0


@pytest.mark.skipif(
    not Path("data/ie_data.xls").exists(),
    reason="Shiller data not present",
)
def test_go_back_year_emits_warning() -> None:
    with pytest.warns(UserWarning, match="go_back_year"):
        run_simulation_mp(
            n_sims=200,
            n_years=15,
            initial_balance=2_000_000,
            withdrawal=400_000,
            withdrawal_negative_year=400_000,
            go_back_year=3,
            random_seed=1,
        )


@pytest.mark.skipif(
    not Path("data/ie_data.xls").exists(),
    reason="Shiller data not present",
)
def test_annual_expense_ratio_reduces_returns() -> None:
    base = run_simulation_mp(
        n_sims=2_000,
        n_years=25,
        initial_balance=2_000_000,
        withdrawal=80_000,
        withdrawal_negative_year=80_000,
        random_seed=7,
        annual_expense_ratio=0.0,
    )
    drag = run_simulation_mp(
        n_sims=2_000,
        n_years=25,
        initial_balance=2_000_000,
        withdrawal=80_000,
        withdrawal_negative_year=80_000,
        random_seed=7,
        annual_expense_ratio=0.01,
    )
    assert drag.final_balances.mean() <= base.final_balances.mean()
