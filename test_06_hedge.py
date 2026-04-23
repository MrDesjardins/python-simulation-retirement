from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from common import HedgingConfig, annual_hedging_drag_estimate, apply_hedge_to_equity_return


MODULE_PATH = Path(__file__).resolve().with_name("06_hedge.py")
SPEC = importlib.util.spec_from_file_location("hedge_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load module spec for {MODULE_PATH}")
hedge_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hedge_module)


def test_none_returns_unchanged() -> None:
    config = HedgingConfig(enabled=False, strategy="none", rebalance_frequency="yearly")
    assert apply_hedge_to_equity_return(-0.12, config, periods_per_year=1) == pytest.approx(-0.12)


def test_protective_put_floor_and_cost_yearly() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="yearly",
        protective_put_floor=-0.15,
        protective_put_cost_annual=0.015,
    )
    # Expected: max(-0.30, -0.15) - 0.015 = -0.165
    assert apply_hedge_to_equity_return(-0.30, config, periods_per_year=1) == pytest.approx(-0.165)


def test_protective_put_partial_coverage_yearly() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="yearly",
        protective_put_floor=-0.15,
        protective_put_cost_annual=0.015,
        protective_put_coverage=0.70,
    )
    # -0.30 - 0.015 + 0.70 * (0.15) = -0.21
    assert apply_hedge_to_equity_return(-0.30, config, periods_per_year=1) == pytest.approx(-0.21)


def test_protective_put_higher_coverage_protects_more() -> None:
    low = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="yearly",
        protective_put_floor=-0.15,
        protective_put_cost_annual=0.015,
        protective_put_coverage=0.30,
    )
    high = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="yearly",
        protective_put_floor=-0.15,
        protective_put_cost_annual=0.015,
        protective_put_coverage=0.70,
    )
    low_out = apply_hedge_to_equity_return(-0.30, low, periods_per_year=1)
    high_out = apply_hedge_to_equity_return(-0.30, high, periods_per_year=1)
    assert high_out > low_out


def test_tail_hedge_only_activates_below_trigger() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="tail_hedge",
        rebalance_frequency="yearly",
        tail_hedge_trigger=-0.20,
        tail_hedge_slope=0.75,
        tail_hedge_cost_annual=0.005,
    )
    # Above trigger: no protection, only cost
    assert apply_hedge_to_equity_return(-0.10, config, periods_per_year=1) == pytest.approx(-0.105)
    # Below trigger: -0.30 - 0.005 + 0.75 * 0.10 = -0.23
    assert apply_hedge_to_equity_return(-0.30, config, periods_per_year=1) == pytest.approx(-0.23)


def test_collar_caps_and_floors() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="collar",
        rebalance_frequency="yearly",
        collar_floor=-0.10,
        collar_cap=0.10,
        collar_cost_annual=0.0,
    )
    assert apply_hedge_to_equity_return(-0.25, config, periods_per_year=1) == pytest.approx(-0.10)
    assert apply_hedge_to_equity_return(0.18, config, periods_per_year=1) == pytest.approx(0.10)


def test_covered_call_caps_upside_on_written_fraction() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="covered_call",
        rebalance_frequency="yearly",
        covered_call_write_fraction=0.10,
        covered_call_strike_otm=0.05,
        covered_call_premium_annual=0.01,
        covered_call_assignment_cost=0.002,
    )
    out = apply_hedge_to_equity_return(0.12, config, periods_per_year=1)
    assert out == pytest.approx(0.1138)


def test_covered_call_keeps_downside_with_small_premium_offset() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="covered_call",
        rebalance_frequency="yearly",
        covered_call_write_fraction=0.10,
        covered_call_strike_otm=0.05,
        covered_call_premium_annual=0.01,
        covered_call_assignment_cost=0.002,
    )
    out = apply_hedge_to_equity_return(-0.10, config, periods_per_year=1)
    assert out == pytest.approx(-0.099)


def test_covered_call_drag_estimate_is_negative_income() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="covered_call",
        rebalance_frequency="yearly",
        covered_call_premium_annual=0.006,
    )
    assert annual_hedging_drag_estimate(config) == pytest.approx(-0.006)


def test_monthly_vs_yearly_cost_scaling() -> None:
    config = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="monthly",
        protective_put_floor=-0.50,
        protective_put_cost_annual=0.012,
    )
    yearly = apply_hedge_to_equity_return(0.00, config, periods_per_year=1)
    monthly = apply_hedge_to_equity_return(0.00, config, periods_per_year=12)

    assert yearly == pytest.approx(-0.012)
    assert monthly == pytest.approx(-0.001)


def test_strategy_order_runs_baseline_then_three_hedges() -> None:
    configs = hedge_module.build_strategy_configs(frequency="yearly")
    assert [c.strategy for c in configs] == [
        "none",
        "protective_put",
        "tail_hedge",
        "collar",
        "covered_call",
    ]


def test_rehedge_substeps_changes_yearly_hedge_shape() -> None:
    no_substeps = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="yearly",
        protective_put_floor=-0.15,
        protective_put_cost_annual=0.015,
        rehedge_substeps_per_period=1,
    )
    with_substeps = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="yearly",
        protective_put_floor=-0.15,
        protective_put_cost_annual=0.015,
        rehedge_substeps_per_period=12,
    )
    yearly_no_substeps = apply_hedge_to_equity_return(-0.30, no_substeps, periods_per_year=1)
    yearly_with_substeps = apply_hedge_to_equity_return(
        -0.30, with_substeps, periods_per_year=1
    )
    assert yearly_no_substeps == pytest.approx(-0.165)
    assert yearly_with_substeps == pytest.approx(-0.1628341673)


def test_add_decision_analytics_baseline_deltas_zero() -> None:
    df = pd.DataFrame(
        [
            {
                "strategy": "none",
                "success_rate": 0.90,
                "failure_rate": 0.10,
                "median_final_balance": 1_000_000.0,
                "cagr": 0.04,
                "max_drawdown": -1.0,
                "avg_annual_hedging_drag_estimate": 0.0,
                "activation_rate": 0.0,
                "floor_hit_rate": 0.0,
                "cap_hit_rate": 0.0,
                "contracts_processed": 1000,
            },
            {
                "strategy": "tail_hedge",
                "success_rate": 0.93,
                "failure_rate": 0.07,
                "median_final_balance": 900_000.0,
                "cagr": 0.035,
                "max_drawdown": -0.8,
                "avg_annual_hedging_drag_estimate": 0.01,
                "activation_rate": 0.2,
                "floor_hit_rate": 0.2,
                "cap_hit_rate": 0.0,
                "contracts_processed": 1000,
            },
        ]
    )
    out = hedge_module.add_decision_analytics(df)
    baseline = out[out["strategy"] == "none"].iloc[0]
    assert baseline["delta_success_rate"] == pytest.approx(0.0)
    assert baseline["delta_failure_rate"] == pytest.approx(0.0)
    assert baseline["delta_cagr"] == pytest.approx(0.0)
    assert baseline["delta_max_drawdown"] == pytest.approx(0.0)
    assert baseline["recommendation"] == "borderline"
    assert baseline["recommendation_reason"] == "baseline_reference"


def test_decision_score_and_ranking_direction() -> None:
    df = pd.DataFrame(
        [
            {
                "strategy": "none",
                "success_rate": 0.90,
                "failure_rate": 0.10,
                "median_final_balance": 1_000_000.0,
                "cagr": 0.04,
                "max_drawdown": -1.0,
                "avg_annual_hedging_drag_estimate": 0.0,
                "activation_rate": 0.0,
                "floor_hit_rate": 0.0,
                "cap_hit_rate": 0.0,
                "contracts_processed": 1000,
            },
            {
                "strategy": "protective_put",
                "success_rate": 0.95,
                "failure_rate": 0.05,
                "median_final_balance": 1_100_000.0,
                "cagr": 0.045,
                "max_drawdown": -0.7,
                "avg_annual_hedging_drag_estimate": 0.01,
                "activation_rate": 0.1,
                "floor_hit_rate": 0.1,
                "cap_hit_rate": 0.0,
                "contracts_processed": 1000,
            },
            {
                "strategy": "collar",
                "success_rate": 0.80,
                "failure_rate": 0.20,
                "median_final_balance": 500_000.0,
                "cagr": 0.01,
                "max_drawdown": -1.0,
                "avg_annual_hedging_drag_estimate": 0.0,
                "activation_rate": 0.3,
                "floor_hit_rate": 0.1,
                "cap_hit_rate": 0.2,
                "contracts_processed": 1000,
            },
        ]
    )
    out = hedge_module.add_decision_analytics(df)
    by_strategy = {r["strategy"]: r for _, r in out.iterrows()}

    assert by_strategy["protective_put"]["worth_it_score"] > by_strategy["none"]["worth_it_score"]
    assert by_strategy["none"]["worth_it_score"] > by_strategy["collar"]["worth_it_score"]
    assert by_strategy["protective_put"]["rank"] == 1
    assert by_strategy["collar"]["recommendation"] == "not_worth_it"


def test_build_hedging_config_from_params_for_tail() -> None:
    params = {
        "tail_hedge_trigger": -0.14,
        "tail_hedge_slope": 0.55,
        "premium_adjustment": 0.0,
    }
    cfg = hedge_module._build_hedging_config_from_params(
        "tail_hedge",
        params,
    )
    assert cfg.strategy == "tail_hedge"
    assert cfg.tail_hedge_trigger == pytest.approx(-0.14)
    assert cfg.tail_hedge_slope == pytest.approx(0.55)
    expected_cost = hedge_module._strategy_cost_model("tail_hedge", params)
    assert cfg.tail_hedge_cost_annual == pytest.approx(expected_cost)


def test_build_hedging_config_from_params_for_protective_put_coverage() -> None:
    params = {
        "protective_put_floor": -0.11,
        "protective_put_coverage": 0.60,
        "premium_adjustment": 0.0,
    }
    cfg = hedge_module._build_hedging_config_from_params("protective_put", params)
    assert cfg.strategy == "protective_put"
    assert cfg.protective_put_floor == pytest.approx(-0.11)
    assert cfg.protective_put_coverage == pytest.approx(0.60)


def test_build_hedging_config_from_params_validates_collar_bounds() -> None:
    with pytest.raises(ValueError):
        hedge_module._build_hedging_config_from_params(
            "collar",
            {
                "collar_floor": 0.06,
                "collar_cap": 0.04,
                "collar_cost_annual": 0.0,
            },
        )


def test_build_hedging_config_from_params_for_covered_call() -> None:
    params = {
        "covered_call_write_fraction": 0.10,
        "covered_call_strike_otm": 0.05,
        "premium_adjustment": 0.0,
    }
    cfg = hedge_module._build_hedging_config_from_params("covered_call", params)
    assert cfg.strategy == "covered_call"
    assert cfg.covered_call_write_fraction == pytest.approx(0.10)
    assert cfg.covered_call_strike_otm == pytest.approx(0.05)


def test_failure_first_objective_score_prefers_lower_failure() -> None:
    better = hedge_module._failure_first_objective_score(
        baseline_failure_rate=0.10,
        strategy_failure_rate=0.06,
        baseline_cagr=0.04,
        strategy_cagr=0.039,
        baseline_max_drawdown=-1.0,
        strategy_max_drawdown=-0.95,
        annual_drag=0.01,
        activation_rate=0.05,
    )
    worse = hedge_module._failure_first_objective_score(
        baseline_failure_rate=0.10,
        strategy_failure_rate=0.14,
        baseline_cagr=0.04,
        strategy_cagr=0.045,
        baseline_max_drawdown=-0.95,
        strategy_max_drawdown=-1.0,
        annual_drag=0.00,
        activation_rate=0.05,
    )
    assert better > worse


def test_robustness_summary_aggregates_ranges() -> None:
    rows = [
        {"failure_rate": 0.08, "success_rate": 0.92, "cagr": 0.03, "worth_it_score": 0.5},
        {"failure_rate": 0.10, "success_rate": 0.90, "cagr": 0.02, "worth_it_score": -1.0},
        {"failure_rate": 0.06, "success_rate": 0.94, "cagr": 0.04, "worth_it_score": 1.5},
    ]
    summary = hedge_module._robustness_summary(rows=rows, strategy="tail_hedge")
    assert summary["strategy"] == "tail_hedge"
    assert summary["failure_rate_min"] == pytest.approx(0.06)
    assert summary["failure_rate_max"] == pytest.approx(0.10)
    assert summary["success_rate_median"] == pytest.approx(0.92)


def test_strategy_cost_model_monotonic_for_put_floor_strength() -> None:
    weak = hedge_module._strategy_cost_model(
        "protective_put",
        {"protective_put_floor": -0.18, "premium_adjustment": 0.0},
    )
    strong = hedge_module._strategy_cost_model(
        "protective_put",
        {"protective_put_floor": -0.05, "premium_adjustment": 0.0},
    )
    assert strong > weak


def test_strategy_cost_model_higher_put_coverage_costs_more() -> None:
    low_cov = hedge_module._strategy_cost_model(
        "protective_put",
        {
            "protective_put_floor": -0.10,
            "protective_put_coverage": 0.30,
            "premium_adjustment": 0.0,
        },
    )
    high_cov = hedge_module._strategy_cost_model(
        "protective_put",
        {
            "protective_put_floor": -0.10,
            "protective_put_coverage": 0.70,
            "premium_adjustment": 0.0,
        },
    )
    assert high_cov > low_cov


def test_strategy_cost_model_covered_call_higher_write_fraction_increases_premium() -> None:
    low_write = hedge_module._strategy_cost_model(
        "covered_call",
        {
            "covered_call_write_fraction": 0.02,
            "covered_call_strike_otm": 0.05,
            "premium_adjustment": 0.0,
        },
    )
    high_write = hedge_module._strategy_cost_model(
        "covered_call",
        {
            "covered_call_write_fraction": 0.15,
            "covered_call_strike_otm": 0.05,
            "premium_adjustment": 0.0,
        },
    )
    assert high_write > low_write


def test_covered_call_write_fraction_candidates_include_20_and_25_percent() -> None:
    assert 0.20 in hedge_module.COVERED_CALL_WRITE_FRACTION_CANDIDATES
    assert 0.25 in hedge_module.COVERED_CALL_WRITE_FRACTION_CANDIDATES


def test_build_walk_forward_folds_non_overlapping_train_test() -> None:
    returns = np.arange(0, 252 * 75, dtype=np.float64) / 10_000.0
    folds = hedge_module.build_walk_forward_folds(
        returns,
        train_years=50,
        test_years=10,
        step_years=10,
    )
    assert len(folds) >= 1
    first = folds[0]
    assert len(first.train_returns) == 50 * 252
    assert len(first.test_returns) == 10 * 252
    assert first.train_returns[-1] != first.test_returns[0]


def test_block_bootstrap_returns_preserves_length() -> None:
    rng = np.random.default_rng(123)
    returns = np.linspace(-0.02, 0.02, 400, dtype=np.float64)
    boot = hedge_module.block_bootstrap_returns(returns, block_days=21, rng=rng)
    assert len(boot) == len(returns)
    assert np.isfinite(boot).all()


def test_column_glossary_contains_directionality() -> None:
    lines = hedge_module._column_glossary_lines(
        ["success_rate", "failure_rate", "worth_it_score"]
    )
    joined = "\n".join(lines)
    assert "higher is better" in joined
    assert "lower is better" in joined


def test_split_tuning_and_holdout_returns_sizes() -> None:
    returns = np.linspace(-0.02, 0.02, 252 * 85, dtype=np.float64)
    tuning, holdout = hedge_module.split_tuning_and_holdout_returns(returns, holdout_years=10)
    assert len(holdout) == 252 * 10
    assert len(tuning) == len(returns) - len(holdout)


def test_plausibility_penalty_detects_suspicious_combo() -> None:
    penalty = hedge_module._plausibility_penalty(
        success_rate=0.9999,
        delta_cagr=0.09,
        annual_drag=0.003,
        activation_rate=0.35,
    )
    assert penalty > 8.0


def test_optuna_columns_compact_includes_put_coverage() -> None:
    original = getattr(hedge_module, "OUTPUT_VIEW")
    setattr(hedge_module, "OUTPUT_VIEW", "compact")
    try:
        df = pd.DataFrame(
            [
                    {
                        "strategy": "protective_put",
                        "protective_put_coverage": 0.7,
                        "covered_call_write_fraction": np.nan,
                        "rank": 1,
                        "recommendation": "worth_it",
                    "success_rate": 0.95,
                    "failure_rate": 0.05,
                    "median_final_balance": 1_000_000.0,
                    "cagr": 0.04,
                    "max_drawdown": -0.7,
                    "avg_annual_hedging_drag_estimate": 0.02,
                    "recommendation_reason": "lower_failure",
                }
            ]
        )
        cols = hedge_module._optuna_columns(df)
        assert "protective_put_coverage" in cols
        assert "covered_call_write_fraction" in cols
        assert "recommendation_reason" in cols
    finally:
        setattr(hedge_module, "OUTPUT_VIEW", original)


def test_fallback_warning_lines_formats_partial_and_full_fallbacks() -> None:
    lines = hedge_module._fallback_warning_lines(
        {
            "protective_put": 3,
            "tail_hedge": 1,
            "collar": 0,
        },
        n_folds=3,
    )
    joined = "\n".join(lines)
    assert "protective_put" in joined
    assert "all folds (3/3)" in joined
    assert "tail_hedge" in joined
    assert "1/3 folds" in joined
    assert "collar" not in joined


def test_trial_completion_summary_lines_formats_counts() -> None:
    lines = hedge_module._trial_completion_summary_lines(
        {
            "protective_put": {"complete": 10, "pruned": 2, "fail": 0},
            "tail_hedge": {"complete": 8, "pruned": 4, "fail": 1},
        }
    )
    joined = "\n".join(lines)
    assert "protective_put" in joined
    assert "complete=10" in joined
    assert "pruned=4" in joined
    assert "failed=1" in joined


def test_best_params_per_put_coverage_from_trials_picks_best_value() -> None:
    dist_floor = optuna.distributions.FloatDistribution(low=-0.14, high=-0.08, step=0.01)
    dist_cov = optuna.distributions.CategoricalDistribution(choices=[0.7, 0.6])
    dist_prem = optuna.distributions.FloatDistribution(low=0.0, high=0.015, step=0.001)

    t1 = optuna.trial.create_trial(
        params={
            "protective_put_floor": -0.10,
            "protective_put_coverage": 0.7,
            "premium_adjustment": 0.003,
        },
        distributions={
            "protective_put_floor": dist_floor,
            "protective_put_coverage": dist_cov,
            "premium_adjustment": dist_prem,
        },
        value=1.0,
    )
    t2 = optuna.trial.create_trial(
        params={
            "protective_put_floor": -0.09,
            "protective_put_coverage": 0.7,
            "premium_adjustment": 0.004,
        },
        distributions={
            "protective_put_floor": dist_floor,
            "protective_put_coverage": dist_cov,
            "premium_adjustment": dist_prem,
        },
        value=2.0,
    )
    t3 = optuna.trial.create_trial(
        params={
            "protective_put_floor": -0.11,
            "protective_put_coverage": 0.6,
            "premium_adjustment": 0.002,
        },
        distributions={
            "protective_put_floor": dist_floor,
            "protective_put_coverage": dist_cov,
            "premium_adjustment": dist_prem,
        },
        value=1.5,
    )

    out = hedge_module._best_params_per_put_coverage_from_trials([t1, t2, t3])
    assert set(out.keys()) == {0.7, 0.6}
    assert out[0.7]["protective_put_floor"] == pytest.approx(-0.09)
    assert out[0.6]["protective_put_floor"] == pytest.approx(-0.11)


def test_default_params_for_protective_put_uses_in_grid_fallback_coverage() -> None:
    params = hedge_module._default_params_for_strategy("protective_put")
    assert params["protective_put_coverage"] == pytest.approx(
        hedge_module.PROTECTIVE_PUT_FALLBACK_COVERAGE
    )
    assert params["protective_put_coverage"] in hedge_module.PROTECTIVE_PUT_COVERAGE_CANDIDATES


def test_contract_event_masks_for_covered_call_assignment() -> None:
    cfg = HedgingConfig(
        enabled=True,
        strategy="covered_call",
        rebalance_frequency="yearly",
        covered_call_strike_otm=0.05,
    )
    raw = np.array([0.01, 0.05, 0.08], dtype=np.float64)
    activation, floor_hit, cap_hit = hedge_module._contract_event_masks(raw, cfg)
    assert activation.tolist() == [False, False, True]
    assert floor_hit.tolist() == [False, False, False]
    assert cap_hit.tolist() == [False, False, True]


def test_new_hedge_entry_cadence_metrics_respect_cap() -> None:
    daily_returns = np.zeros(252, dtype=np.float64)
    cfg = HedgingConfig(
        enabled=True,
        strategy="protective_put",
        rebalance_frequency="yearly",
    )
    out = hedge_module.summarize_simulation(
        hedging_config=cfg,
        daily_returns=daily_returns,
        random_seed=123,
        n_sims=100,
    )

    assert out["reset_policy"] == hedge_module.HEDGE_RESET_POLICY
    assert out["entry_cap_days"] == hedge_module.HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES
    assert out["avg_new_hedges_per_sim_per_year"] <= out["max_possible_new_hedges_per_year_from_cap"]
    assert out["avg_new_hedges_per_sim_per_year"] == pytest.approx(hedge_module.CONTRACTS_PER_YEAR)


def test_contract_period_accounting_and_no_hedge_entries_for_none() -> None:
    daily_returns = np.zeros(252, dtype=np.float64)
    out = hedge_module.summarize_simulation(
        hedging_config=HedgingConfig(enabled=False, strategy="none", rebalance_frequency="yearly"),
        daily_returns=daily_returns,
        random_seed=321,
        n_sims=50,
    )
    expected_periods = 50 * hedge_module.RETIREMENT_YEARS * hedge_module.CONTRACTS_PER_YEAR
    assert out["contract_periods_evaluated_total"] == expected_periods
    assert out["contracts_processed"] == expected_periods
    assert out["new_hedges_opened_total"] == 0
    assert out["avg_new_hedges_per_sim_per_year"] == pytest.approx(0.0)
