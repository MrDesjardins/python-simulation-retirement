"""Daily hedge strategy comparison runner using SPX.csv.

This module uses a return-shaping hedge abstraction with periodic 21-day contract
settlement and wait-for-expiry resets. Hedge floor/trigger/cap thresholds are modeled
at the 21-day contract horizon. It does NOT perform options contract pricing,
Greeks, implied volatility surface modeling, or execution-level broker mechanics.
"""

from __future__ import annotations

import json
import os
import time
from typing import Literal, NamedTuple

import numpy as np
import optuna
import pandas as pd

from common import HedgingConfig, annual_hedging_drag_estimate

# Core simulation inputs
SPX_DAILY_FILE_PATH = "data/SPX.csv"
INITIAL_BALANCE = 4_000_000
WITHDRAWAL_POSITIVE_YEAR = 130_000
WITHDRAWAL_NEGATIVE_YEAR = 120_000
N_SIMS = 10_000
RETIREMENT_YEARS = 40
SP500_PERCENTAGE = 0.70
BOND_RATE = 0.036  # Unified with other simulations; nominal, 0% real at equal inflation
INFLATION_RATE = 0.036  # Unified with other simulations
SIMULATION_RANDOM_SEED: int | None = None

YEARS_WITHOUT_SOCIAL_SECURITY = 18
SOCIAL_SECURITY_MONEY = 35_000
YEARS_WITH_SUPPLEMENTAL_INCOME = 10
SUPPLEMENTAL_INCOME = 18_000

SIMULATION_FREQUENCY: Literal["monthly", "yearly"] = "yearly"
TRADING_DAYS_PER_YEAR = 252
OPTION_DAYS = 21
HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES = 5
HEDGE_RESET_POLICY: Literal["wait_for_expiry"] = "wait_for_expiry"
CONTRACTS_PER_YEAR = TRADING_DAYS_PER_YEAR // OPTION_DAYS
if CONTRACTS_PER_YEAR * OPTION_DAYS != TRADING_DAYS_PER_YEAR:
    raise ValueError("TRADING_DAYS_PER_YEAR must be divisible by OPTION_DAYS")
if OPTION_DAYS < 1:
    raise ValueError("OPTION_DAYS must be >= 1")
if HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES < 1:
    raise ValueError("HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES must be >= 1")
if HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES > OPTION_DAYS:
    raise ValueError(
        "HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES cannot exceed OPTION_DAYS for "
        "wait_for_expiry lifecycle in this abstraction."
    )

# Worth-it scoring policy (failure reduction is primary objective)
FAILURE_REDUCTION_WEIGHT = 0.60
DRAWDOWN_IMPROVEMENT_WEIGHT = 0.15
CAGR_WEIGHT = 0.10
HEDGE_DRAG_WEIGHT = 0.15
WORTH_IT_THRESHOLD = 1.00
BORDERLINE_THRESHOLD = 0.00

# Optuna experimentation configuration
RUN_MODE: Literal["comparison", "optuna"] = "optuna"
OUTPUT_VIEW: Literal["compact", "full"] = "compact"
SHOW_GLOSSARY_IN_COMPACT = False
OPTUNA_TRIAL_COUNT_PER_STRATEGY = 180
OPTUNA_STORAGE_PATH = "sqlite:///db_06_hedge_optuna.sqlite3"
OPTUNA_STUDY_PREFIX = "hedge_optuna_v15" ############################# TO CHANGE EVERY TIME WE CHANGE THE STUDY ⚠️
OPTUNA_N_SIMS_SMALL = 1_000
OPTUNA_N_SIMS_LARGE = 3_000
OPTUNA_MIN_ACCEPTABLE_FAILURE_RATE = 0.80
OPTUNA_PRUNER_STARTUP_TRIALS = 20
OPTUNA_PRUNER_WARMUP_STEPS = 1
OPTUNA_EARLY_PRUNE_FAILURE_MARGIN = 0.04
OPTUNA_ENABLE_PRUNING = False
OPTUNA_HOLDOUT_YEARS = 10
OPTUNA_MAX_REASONABLE_CAGR_DELTA = 0.05
OPTUNA_NEAR_PERFECT_SUCCESS_RATE = 0.999
OPTUNA_MIN_DRAG_FOR_NEAR_PERFECT_SUCCESS = 0.015
OPTUNA_MAX_REASONABLE_ACTIVATION_RATE = 0.30
OPTUNA_OBJECTIVE_DRAG_WEIGHT = 0.35
OPTUNA_OBJECTIVE_ACTIVATION_WEIGHT = 0.06
OPTUNA_PRUNE_PENALTY_THRESHOLD_SMALL = 20.0
OPTUNA_PRUNE_PENALTY_THRESHOLD_LARGE = 25.0
PROTECTIVE_PUT_COVERAGE_CANDIDATES: tuple[float, ...] = (0.70, 0.60, 0.50, 0.40, 0.30)
PROTECTIVE_PUT_FALLBACK_COVERAGE = 0.50
COVERED_CALL_WRITE_FRACTION_CANDIDATES: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25)
COVERED_CALL_STRIKE_OTM_CANDIDATES: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15)

ROBUSTNESS_SEEDS = [7, 42, 314159, 271828, 2025]
ROBUSTNESS_N_SIMS = 20_000
RESULTS_OPTUNA_PATH = f"results/{OPTUNA_STUDY_PREFIX}_summary.json"

# Credibility settings
SIMULATION_SAMPLE_MODE: Literal["iid", "block_bootstrap"] = "iid"
SIMULATION_BLOCK_DAYS = 63
WF_TRAIN_YEARS = 50
WF_TEST_YEARS = 10
WF_STEP_YEARS = 10
WF_VALIDATION_N_SIMS = 20_000
BOOTSTRAP_DRAWS = 4
BOOTSTRAP_N_SIMS = 6_000
BOOTSTRAP_BLOCK_DAYS = 63


class WalkForwardFold(NamedTuple):
    fold_id: int
    train_returns: np.ndarray
    test_returns: np.ndarray


def load_daily_returns_from_spx_csv(file_path: str = SPX_DAILY_FILE_PATH) -> np.ndarray:
    """Build daily close-to-close returns from SPX daily history."""
    df = pd.read_csv(file_path)
    required_columns = {"Date", "Adj Close", "Close"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date")
    price_col = "Adj Close" if df["Adj Close"].notna().any() else "Close"
    prices = df[price_col].dropna()
    if len(prices) < 2:
        raise ValueError(f"{file_path} has insufficient daily prices")
    daily_returns = prices.pct_change().dropna().to_numpy(dtype=np.float64)
    if len(daily_returns) == 0:
        raise ValueError(f"{file_path} did not produce daily returns")
    return daily_returns


def build_walk_forward_folds(
    daily_returns: np.ndarray,
    *,
    train_years: int = WF_TRAIN_YEARS,
    test_years: int = WF_TEST_YEARS,
    step_years: int = WF_STEP_YEARS,
) -> list[WalkForwardFold]:
    train_days = train_years * TRADING_DAYS_PER_YEAR
    test_days = test_years * TRADING_DAYS_PER_YEAR
    step_days = step_years * TRADING_DAYS_PER_YEAR
    n = len(daily_returns)
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("walk-forward day parameters must be positive")
    folds: list[WalkForwardFold] = []
    start = 0
    fold_id = 0
    while start + train_days + test_days <= n:
        train_slice = daily_returns[start : start + train_days]
        test_slice = daily_returns[start + train_days : start + train_days + test_days]
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_returns=np.asarray(train_slice, dtype=np.float64),
                test_returns=np.asarray(test_slice, dtype=np.float64),
            )
        )
        fold_id += 1
        start += step_days
    if not folds:
        raise ValueError(
            "Not enough data for walk-forward folds. "
            f"Need at least {train_days + test_days} daily returns, got {n}."
        )
    return folds


def split_tuning_and_holdout_returns(
    daily_returns: np.ndarray,
    *,
    holdout_years: int = OPTUNA_HOLDOUT_YEARS,
) -> tuple[np.ndarray, np.ndarray]:
    holdout_days = holdout_years * TRADING_DAYS_PER_YEAR
    if holdout_days <= 0:
        raise ValueError(f"holdout_years must be positive, got {holdout_years}")

    min_tuning_days = (WF_TRAIN_YEARS + WF_TEST_YEARS) * TRADING_DAYS_PER_YEAR
    if len(daily_returns) <= holdout_days + min_tuning_days:
        raise ValueError(
            "Not enough data for holdout split. "
            f"Need > {holdout_days + min_tuning_days} days, got {len(daily_returns)}."
        )

    tuning = np.asarray(daily_returns[:-holdout_days], dtype=np.float64)
    holdout = np.asarray(daily_returns[-holdout_days:], dtype=np.float64)
    return tuning, holdout


def block_bootstrap_returns(
    returns: np.ndarray,
    *,
    block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_days <= 0:
        raise ValueError(f"block_days must be positive, got {block_days}")
    n = len(returns)
    if n == 0:
        raise ValueError("returns cannot be empty")
    out = np.empty(n, dtype=np.float64)
    write_pos = 0
    max_start = max(1, n - block_days + 1)
    while write_pos < n:
        start = int(rng.integers(0, max_start))
        end = min(start + block_days, n)
        block = returns[start:end]
        take = min(len(block), n - write_pos)
        out[write_pos : write_pos + take] = block[:take]
        write_pos += take
    return out


def _sample_year_returns_block_bootstrap(
    *,
    n_sims: int,
    days_per_year: int,
    returns: np.ndarray,
    block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_days <= 0:
        raise ValueError(f"block_days must be positive, got {block_days}")
    n = len(returns)
    if n == 0:
        raise ValueError("returns cannot be empty")

    blocks_per_row = (days_per_year + block_days - 1) // block_days
    max_start = max(1, n - block_days + 1)
    starts = rng.integers(0, max_start, size=(n_sims, blocks_per_row))
    offsets = np.arange(block_days, dtype=np.int64)
    idx = starts[..., None] + offsets
    idx = np.minimum(idx, n - 1)
    sampled = returns[idx].reshape(n_sims, blocks_per_row * block_days)
    return sampled[:, :days_per_year]


def _strategy_cost_model(
    strategy: Literal["protective_put", "tail_hedge", "collar", "covered_call"],
    params: dict[str, float],
) -> float:
    if strategy == "protective_put":
        floor = float(params["protective_put_floor"])
        coverage = float(params.get("protective_put_coverage", 1.0))
        premium_adj = float(params.get("premium_adjustment", 0.0))
        coverage_strength = max(0.0, min(1.0, coverage))
        protection_strength = max(0.0, min(0.14, 0.16 + floor))
        cost = 0.010 + 0.28 * protection_strength + 0.025 * coverage_strength + premium_adj
        return float(np.clip(cost, 0.015, 0.09))

    if strategy == "tail_hedge":
        trigger = float(params["tail_hedge_trigger"])
        slope = float(params["tail_hedge_slope"])
        premium_adj = float(params.get("premium_adjustment", 0.0))
        trigger_strength = max(0.0, min(0.13, 0.22 + trigger))
        slope_strength = max(0.0, min(1.0, (slope - 0.25) / 0.55))
        cost = 0.004 + 0.08 * trigger_strength + 0.035 * slope_strength + premium_adj
        return float(np.clip(cost, 0.003, 0.05))

    if strategy == "collar":
        floor = float(params["collar_floor"])
        cap = float(params["collar_cap"])
        premium_adj = float(params.get("premium_adjustment", 0.0))
        floor_strength = max(0.0, min(0.10, 0.12 + floor))
        cap_tightness = max(0.0, min(0.07, 0.10 - cap))
        put_leg = 0.012 + 0.28 * floor_strength
        call_credit = 0.003 + 0.18 * cap_tightness
        net = put_leg - call_credit + premium_adj
        if floor > -0.07 and cap > 0.08:
            net = max(net, 0.0)
        return float(np.clip(net, 0.0, 0.05))

    if strategy == "covered_call":
        write_fraction = float(params["covered_call_write_fraction"])
        strike_otm = float(params["covered_call_strike_otm"])
        premium_adj = float(params.get("premium_adjustment", 0.0))
        write_strength = max(0.0, min(0.25, write_fraction)) / 0.25
        strike_tightness = max(0.0, min(0.13, 0.15 - strike_otm)) / 0.13
        premium = 0.0008 + 0.0045 * write_strength + 0.0065 * strike_tightness + premium_adj
        return float(np.clip(premium, 0.0005, 0.015))

    raise ValueError(f"Unsupported strategy for cost model: {strategy}")


def _compute_cagr(median_final_balance: float, initial_balance: float, n_years: int) -> float:
    if median_final_balance <= 0 or initial_balance <= 0 or n_years <= 0:
        return -1.0
    return (median_final_balance / initial_balance) ** (1.0 / n_years) - 1.0


def _max_new_hedges_per_year_from_policy() -> int:
    cadence_days = max(OPTION_DAYS, HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES)
    return TRADING_DAYS_PER_YEAR // cadence_days


def _apply_contract_hedge_vectorized(
    raw_contract_return: np.ndarray,
    hedging_config: HedgingConfig,
    contracts_per_year: int,
) -> np.ndarray:
    """
    Apply hedge payoff approximation on a single contract horizon return (21 days).
    """
    if not hedging_config.enabled or hedging_config.strategy == "none":
        return raw_contract_return

    if hedging_config.strategy == "protective_put":
        # Treat floor as a contract-horizon floor (21-trading-day here).
        # Coverage controls how much downside below the floor is offset.
        floor_contract = hedging_config.protective_put_floor
        coverage = float(np.clip(hedging_config.protective_put_coverage, 0.0, 1.0))
        cost_contract = hedging_config.protective_put_cost_annual / contracts_per_year
        hedged = raw_contract_return - cost_contract
        mask = raw_contract_return < floor_contract
        if np.any(mask):
            hedged = hedged.copy()
            hedged[mask] += coverage * (floor_contract - raw_contract_return[mask])
        return hedged

    if hedging_config.strategy == "tail_hedge":
        # Treat trigger as contract-horizon trigger.
        trigger_contract = hedging_config.tail_hedge_trigger
        cost_contract = hedging_config.tail_hedge_cost_annual / contracts_per_year
        hedged = raw_contract_return - cost_contract
        mask = raw_contract_return < trigger_contract
        if np.any(mask):
            excess_drawdown = trigger_contract - raw_contract_return[mask]
            hedged = hedged.copy()
            hedged[mask] += hedging_config.tail_hedge_slope * excess_drawdown
        return hedged

    if hedging_config.strategy == "collar":
        # Treat floor/cap as contract-horizon bounds.
        floor_contract = hedging_config.collar_floor
        cap_contract = hedging_config.collar_cap
        cost_contract = hedging_config.collar_cost_annual / contracts_per_year
        bounded = np.minimum(cap_contract, np.maximum(raw_contract_return, floor_contract))
        return bounded - cost_contract

    if hedging_config.strategy == "covered_call":
        write_fraction = float(np.clip(hedging_config.covered_call_write_fraction, 0.0, 1.0))
        strike_contract = hedging_config.covered_call_strike_otm
        premium_contract = hedging_config.covered_call_premium_annual / contracts_per_year
        assignment_cost = hedging_config.covered_call_assignment_cost
        written_leg = premium_contract + np.minimum(raw_contract_return, strike_contract)
        assigned_mask = raw_contract_return > strike_contract
        if np.any(assigned_mask):
            written_leg = written_leg.copy()
            written_leg[assigned_mask] -= assignment_cost
        return (1.0 - write_fraction) * raw_contract_return + write_fraction * written_leg

    raise ValueError(f"Unsupported hedging strategy: {hedging_config.strategy}")


def _contract_event_masks(
    raw_contract_return: np.ndarray,
    hedging_config: HedgingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return masks for (activation, floor_hit, cap_hit) at contract horizon.
    """
    n = raw_contract_return.shape[0]
    zeros = np.zeros(n, dtype=bool)
    if not hedging_config.enabled or hedging_config.strategy == "none":
        return zeros, zeros, zeros

    if hedging_config.strategy == "protective_put":
        floor_hit = raw_contract_return < hedging_config.protective_put_floor
        return floor_hit, floor_hit, zeros

    if hedging_config.strategy == "tail_hedge":
        trigger_hit = raw_contract_return < hedging_config.tail_hedge_trigger
        return trigger_hit, trigger_hit, zeros

    if hedging_config.strategy == "collar":
        floor_hit = raw_contract_return < hedging_config.collar_floor
        cap_hit = raw_contract_return > hedging_config.collar_cap
        activation = floor_hit | cap_hit
        return activation, floor_hit, cap_hit

    if hedging_config.strategy == "covered_call":
        assigned = raw_contract_return > hedging_config.covered_call_strike_otm
        return assigned, zeros, assigned

    raise ValueError(f"Unsupported hedging strategy: {hedging_config.strategy}")


def build_strategy_configs(
    frequency: Literal["monthly", "yearly"] = SIMULATION_FREQUENCY,
) -> list[HedgingConfig]:
    # These settings are calibrated for a 21-trading-day contract horizon.
    # Costs are annual drags and are pro-rated per contract in simulation.
    # This remains a simplified return-shaping model, not executable options P&L.
    return [
        HedgingConfig(enabled=False, strategy="none", rebalance_frequency=frequency),
        HedgingConfig(
            enabled=True,
            strategy="protective_put",
            rebalance_frequency=frequency,
            protective_put_floor=-0.10,
            protective_put_cost_annual=0.03,
            protective_put_coverage=1.0,
        ),
        HedgingConfig(
            enabled=True,
            strategy="tail_hedge",
            rebalance_frequency=frequency,
            tail_hedge_trigger=-0.12,
            tail_hedge_slope=0.50,
            tail_hedge_cost_annual=0.01,
        ),
        HedgingConfig(
            enabled=True,
            strategy="collar",
            rebalance_frequency=frequency,
            collar_floor=-0.08,
            collar_cap=0.05,
            collar_cost_annual=0.00,
        ),
        HedgingConfig(
            enabled=True,
            strategy="covered_call",
            rebalance_frequency=frequency,
            covered_call_write_fraction=0.05,
            covered_call_strike_otm=0.10,
            covered_call_premium_annual=0.004,
            covered_call_assignment_cost=0.002,
        ),
    ]


def _social_security_by_year(n_years: int) -> np.ndarray:
    inflation_factors = (1.0 + INFLATION_RATE) ** np.arange(n_years)
    ss = np.zeros(n_years, dtype=np.float64)
    if SOCIAL_SECURITY_MONEY > 0:
        years_with_ss = max(0, n_years - YEARS_WITHOUT_SOCIAL_SECURITY)
        if years_with_ss > 0:
            ss[-years_with_ss:] = SOCIAL_SECURITY_MONEY * inflation_factors[-years_with_ss:]
    return ss


def _supplemental_by_year(n_years: int) -> np.ndarray:
    inflation_factors = (1.0 + INFLATION_RATE) ** np.arange(n_years)
    supp = np.zeros(n_years, dtype=np.float64)
    if SUPPLEMENTAL_INCOME > 0:
        years_with_supplemental = min(YEARS_WITH_SUPPLEMENTAL_INCOME, n_years)
        supp[:years_with_supplemental] = (
            SUPPLEMENTAL_INCOME * inflation_factors[:years_with_supplemental]
        )
    return supp


def summarize_simulation(
    *,
    hedging_config: HedgingConfig,
    daily_returns: np.ndarray,
    random_seed: int | None,
    n_sims: int = N_SIMS,
    sample_mode: Literal["iid", "block_bootstrap"] = SIMULATION_SAMPLE_MODE,
    block_days: int = SIMULATION_BLOCK_DAYS,
) -> dict[str, float | str | None]:
    n_days_total = RETIREMENT_YEARS * TRADING_DAYS_PER_YEAR
    m = len(daily_returns)
    if n_days_total <= 0:
        raise ValueError("n_days_total must be positive")

    ss_by_year = _social_security_by_year(RETIREMENT_YEARS)
    supp_by_year = _supplemental_by_year(RETIREMENT_YEARS)
    inflation_factors = (1.0 + INFLATION_RATE) ** np.arange(RETIREMENT_YEARS)

    bond_daily_growth = (1.0 + BOND_RATE) ** (1.0 / TRADING_DAYS_PER_YEAR)
    bond_contract_growth = bond_daily_growth**OPTION_DAYS

    final_balances_chunks: list[np.ndarray] = []
    global_worst_drawdown = 0.0
    block_return_sum = 0.0
    block_return_sq_sum = 0.0
    block_return_count = 0
    contract_periods_evaluated_total = 0
    new_hedges_opened_total = 0
    activation_count = 0
    floor_hit_count = 0
    cap_hit_count = 0
    assignment_count = 0

    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")

    remaining = n_sims
    chunk_size = 20_000
    seed_base = (
        int(np.random.default_rng().integers(0, 2**31 - 1))
        if random_seed is None
        else int(random_seed)
    )
    chunk_index = 0

    while remaining > 0:
        c = min(chunk_size, remaining)
        remaining -= c

        rng = np.random.default_rng(seed_base + chunk_index)
        chunk_index += 1

        equity = np.full(c, INITIAL_BALANCE * SP500_PERCENTAGE, dtype=np.float64)
        bond = np.full(c, INITIAL_BALANCE * (1.0 - SP500_PERCENTAGE), dtype=np.float64)
        peak = equity + bond
        worst_drawdown_chunk = np.zeros(c, dtype=np.float64)

        for year in range(RETIREMENT_YEARS):
            if sample_mode == "iid":
                idx = rng.integers(0, m, size=(c, TRADING_DAYS_PER_YEAR))
                year_daily_returns = daily_returns[idx]
            elif sample_mode == "block_bootstrap":
                year_daily_returns = _sample_year_returns_block_bootstrap(
                    n_sims=c,
                    days_per_year=TRADING_DAYS_PER_YEAR,
                    returns=daily_returns,
                    block_days=block_days,
                    rng=rng,
                )
            else:
                raise ValueError(f"Unsupported sample_mode: {sample_mode}")
            year_growth = 1.0 + year_daily_returns
            contract_growth = year_growth.reshape(c, CONTRACTS_PER_YEAR, OPTION_DAYS).prod(axis=2)
            annual_raw_return = contract_growth.prod(axis=1) - 1.0

            withdrawals = np.where(
                annual_raw_return >= 0,
                float(WITHDRAWAL_POSITIVE_YEAR),
                float(WITHDRAWAL_NEGATIVE_YEAR),
            )
            withdrawals *= inflation_factors[year]
            withdrawals = np.maximum(withdrawals - ss_by_year[year] - supp_by_year[year], 0.0)

            total_before_withdrawal = equity + bond
            total_after_withdrawal = np.maximum(total_before_withdrawal - withdrawals, 0.0)
            scale = np.divide(
                total_after_withdrawal,
                np.maximum(total_before_withdrawal, 1e-12),
            )
            equity *= scale
            bond *= scale

            for contract_i in range(CONTRACTS_PER_YEAR):
                total_before_block = equity + bond
                raw_contract_return = contract_growth[:, contract_i] - 1.0
                activation_mask, floor_hit_mask, cap_hit_mask = _contract_event_masks(
                    raw_contract_return,
                    hedging_config,
                )
                contract_periods_evaluated_total += c
                if hedging_config.enabled and hedging_config.strategy != "none":
                    new_hedges_opened_total += c
                activation_count += int(np.sum(activation_mask))
                floor_hit_count += int(np.sum(floor_hit_mask))
                cap_hit_count += int(np.sum(cap_hit_mask))
                if hedging_config.strategy == "covered_call":
                    assignment_count += int(np.sum(cap_hit_mask))

                if hedging_config.apply_to_equity_only:
                    hedged_contract_return = _apply_contract_hedge_vectorized(
                        raw_contract_return,
                        hedging_config,
                        CONTRACTS_PER_YEAR,
                    )
                    equity *= np.maximum(1.0 + hedged_contract_return, 0.0)
                    bond *= bond_contract_growth
                else:
                    blended_raw_contract_return = (
                        SP500_PERCENTAGE * raw_contract_return
                        + (1.0 - SP500_PERCENTAGE) * (bond_contract_growth - 1.0)
                    )
                    hedged_total_return = _apply_contract_hedge_vectorized(
                        blended_raw_contract_return,
                        hedging_config,
                        CONTRACTS_PER_YEAR,
                    )
                    total_after_block = np.maximum(
                        total_before_block * (1.0 + hedged_total_return),
                        0.0,
                    )
                    total_after_block_1d = np.asarray(
                        total_after_block,
                        dtype=np.float64,
                    ).reshape(-1)
                    equity[...] = total_after_block_1d * SP500_PERCENTAGE
                    bond[...] = total_after_block_1d * (1.0 - SP500_PERCENTAGE)

                total_after_block = equity + bond
                np.maximum(total_after_block, 0.0, out=total_after_block)

                block_returns = np.divide(
                    total_after_block,
                    np.maximum(total_before_block, 1e-12),
                ) - 1.0
                block_return_sum += float(np.sum(block_returns))
                block_return_sq_sum += float(np.sum(block_returns * block_returns))
                block_return_count += c

                peak = np.maximum(peak, total_after_block)
                drawdown = np.divide(
                    total_after_block,
                    np.maximum(peak, 1e-12),
                ) - 1.0
                worst_drawdown_chunk = np.minimum(worst_drawdown_chunk, drawdown)

        final_balances = np.maximum(equity + bond, 0.0)
        final_balances_chunks.append(final_balances)
        global_worst_drawdown = min(global_worst_drawdown, float(np.min(worst_drawdown_chunk)))

    final_balances_all = np.concatenate(final_balances_chunks)
    success_rate = float(np.mean(final_balances_all > 0))
    failure_rate = 1.0 - success_rate
    median_final_balance = float(np.median(final_balances_all))
    cagr = _compute_cagr(median_final_balance, INITIAL_BALANCE, RETIREMENT_YEARS)

    if block_return_count > 1:
        mean_block = block_return_sum / block_return_count
        var_block = (block_return_sq_sum / block_return_count) - (mean_block**2)
        block_std = float(np.sqrt(max(var_block, 0.0)))
        volatility = block_std * np.sqrt(CONTRACTS_PER_YEAR)
    else:
        volatility = None

    expected_contract_periods = n_sims * RETIREMENT_YEARS * CONTRACTS_PER_YEAR
    if contract_periods_evaluated_total != expected_contract_periods:
        raise ValueError(
            "Contract period accounting mismatch: "
            f"expected={expected_contract_periods}, "
            f"actual={contract_periods_evaluated_total}"
        )

    activation_rate = (
        activation_count / contract_periods_evaluated_total
        if contract_periods_evaluated_total > 0
        else 0.0
    )
    floor_hit_rate = (
        floor_hit_count / contract_periods_evaluated_total
        if contract_periods_evaluated_total > 0
        else 0.0
    )
    cap_hit_rate = (
        cap_hit_count / contract_periods_evaluated_total
        if contract_periods_evaluated_total > 0
        else 0.0
    )
    avg_new_hedges_per_sim_per_year = (
        new_hedges_opened_total / (n_sims * RETIREMENT_YEARS)
        if n_sims > 0 and RETIREMENT_YEARS > 0
        else 0.0
    )
    max_possible_new_hedges_per_year_from_cap = _max_new_hedges_per_year_from_policy()
    assignment_rate = (
        assignment_count / contract_periods_evaluated_total
        if contract_periods_evaluated_total > 0
        else 0.0
    )
    avg_assignments_per_sim_per_year = (
        assignment_count / (n_sims * RETIREMENT_YEARS)
        if n_sims > 0 and RETIREMENT_YEARS > 0
        else 0.0
    )

    return {
        "strategy": hedging_config.strategy,
        "covered_call_write_fraction": (
            float(hedging_config.covered_call_write_fraction)
            if hedging_config.strategy == "covered_call" and hedging_config.enabled
            else None
        ),
        "covered_call_strike_otm": (
            float(hedging_config.covered_call_strike_otm)
            if hedging_config.strategy == "covered_call" and hedging_config.enabled
            else None
        ),
        "protective_put_coverage": (
            float(hedging_config.protective_put_coverage)
            if hedging_config.strategy == "protective_put" and hedging_config.enabled
            else None
        ),
        "n_sims": int(n_sims),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "median_final_balance": median_final_balance,
        "cagr": cagr,
        "max_drawdown": global_worst_drawdown,
        "volatility": volatility,
        "avg_annual_hedging_drag_estimate": annual_hedging_drag_estimate(hedging_config),
        "contracts_processed": contract_periods_evaluated_total,
        "contract_periods_evaluated_total": contract_periods_evaluated_total,
        "new_hedges_opened_total": new_hedges_opened_total,
        "avg_new_hedges_per_sim_per_year": avg_new_hedges_per_sim_per_year,
        "max_possible_new_hedges_per_year_from_cap": max_possible_new_hedges_per_year_from_cap,
        "entry_cap_days": HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES,
        "reset_policy": HEDGE_RESET_POLICY,
        "activation_rate": activation_rate,
        "floor_hit_rate": floor_hit_rate,
        "cap_hit_rate": cap_hit_rate,
        "assignment_rate": assignment_rate,
        "assignment_count_total": assignment_count,
        "avg_assignments_per_sim_per_year": avg_assignments_per_sim_per_year,
    }


def _recommendation_for_score(score: float) -> str:
    if score >= WORTH_IT_THRESHOLD:
        return "worth_it"
    if score >= BORDERLINE_THRESHOLD:
        return "borderline"
    return "not_worth_it"


def add_decision_analytics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    baseline_rows = out[out["strategy"] == "none"]
    if baseline_rows.empty:
        raise ValueError("Expected baseline strategy 'none' in comparison output")
    baseline = baseline_rows.iloc[0]

    out["delta_success_rate"] = out["success_rate"] - float(baseline["success_rate"])
    out["delta_failure_rate"] = out["failure_rate"] - float(baseline["failure_rate"])
    out["delta_cagr"] = out["cagr"] - float(baseline["cagr"])
    out["delta_median_final_balance"] = (
        out["median_final_balance"] - float(baseline["median_final_balance"])
    )
    out["delta_max_drawdown"] = out["max_drawdown"] - float(baseline["max_drawdown"])

    failure_reduction = float(baseline["failure_rate"]) - out["failure_rate"]
    drawdown_improvement = out["max_drawdown"] - float(baseline["max_drawdown"])
    cagr_delta = out["cagr"] - float(baseline["cagr"])
    annual_drag = out["avg_annual_hedging_drag_estimate"]

    out["worth_it_score"] = 100.0 * (
        FAILURE_REDUCTION_WEIGHT * failure_reduction
        + DRAWDOWN_IMPROVEMENT_WEIGHT * drawdown_improvement
        + CAGR_WEIGHT * cagr_delta
        - HEDGE_DRAG_WEIGHT * annual_drag
    )
    out["recommendation"] = out["worth_it_score"].map(_recommendation_for_score)
    out["rank"] = out["worth_it_score"].rank(method="min", ascending=False).astype(int)

    def reason_for_row(row: pd.Series) -> str:
        if row["strategy"] == "none":
            return "baseline_reference"
        reason_parts: list[str] = []
        if row["delta_failure_rate"] < 0:
            reason_parts.append("lower_failure")
        elif row["delta_failure_rate"] > 0:
            reason_parts.append("higher_failure")
        if row["delta_cagr"] > 0:
            reason_parts.append("higher_cagr")
        elif row["delta_cagr"] < 0:
            reason_parts.append("cagr_drag")
        if row["avg_annual_hedging_drag_estimate"] > 0:
            reason_parts.append("premium_cost")
        if row["activation_rate"] > 0:
            reason_parts.append("active_usage")
        return ",".join(reason_parts)

    out["recommendation_reason"] = out.apply(reason_for_row, axis=1)
    return out


def _default_strategy_config_map(
    frequency: Literal["monthly", "yearly"] = SIMULATION_FREQUENCY,
) -> dict[str, HedgingConfig]:
    return {cfg.strategy: cfg for cfg in build_strategy_configs(frequency=frequency)}


def _build_hedging_config_from_params(
    strategy: Literal["protective_put", "tail_hedge", "collar", "covered_call"],
    params: dict[str, float],
    frequency: Literal["monthly", "yearly"] = SIMULATION_FREQUENCY,
) -> HedgingConfig:
    defaults = _default_strategy_config_map(frequency=frequency)
    base = defaults[strategy]
    if strategy == "protective_put":
        cost = _strategy_cost_model(strategy, params)
        return HedgingConfig(
            enabled=True,
            strategy="protective_put",
            rebalance_frequency=frequency,
            protective_put_floor=float(params["protective_put_floor"]),
            protective_put_cost_annual=cost,
            protective_put_coverage=float(params["protective_put_coverage"]),
            tail_hedge_cost_annual=base.tail_hedge_cost_annual,
            tail_hedge_trigger=base.tail_hedge_trigger,
            tail_hedge_slope=base.tail_hedge_slope,
            collar_floor=base.collar_floor,
            collar_cap=base.collar_cap,
            collar_cost_annual=base.collar_cost_annual,
            covered_call_write_fraction=base.covered_call_write_fraction,
            covered_call_strike_otm=base.covered_call_strike_otm,
            covered_call_premium_annual=base.covered_call_premium_annual,
            covered_call_assignment_cost=base.covered_call_assignment_cost,
        )
    if strategy == "tail_hedge":
        cost = _strategy_cost_model(strategy, params)
        return HedgingConfig(
            enabled=True,
            strategy="tail_hedge",
            rebalance_frequency=frequency,
            protective_put_cost_annual=base.protective_put_cost_annual,
            protective_put_floor=base.protective_put_floor,
            protective_put_coverage=base.protective_put_coverage,
            tail_hedge_cost_annual=cost,
            tail_hedge_trigger=float(params["tail_hedge_trigger"]),
            tail_hedge_slope=float(params["tail_hedge_slope"]),
            collar_floor=base.collar_floor,
            collar_cap=base.collar_cap,
            collar_cost_annual=base.collar_cost_annual,
            covered_call_write_fraction=base.covered_call_write_fraction,
            covered_call_strike_otm=base.covered_call_strike_otm,
            covered_call_premium_annual=base.covered_call_premium_annual,
            covered_call_assignment_cost=base.covered_call_assignment_cost,
        )
    if strategy == "collar":
        floor = float(params["collar_floor"])
        cap = float(params["collar_cap"])
        if floor >= cap:
            raise ValueError(f"collar_floor must be < collar_cap, got {floor} >= {cap}")
        cost = _strategy_cost_model(strategy, params)
        return HedgingConfig(
            enabled=True,
            strategy="collar",
            rebalance_frequency=frequency,
            protective_put_cost_annual=base.protective_put_cost_annual,
            protective_put_floor=base.protective_put_floor,
            protective_put_coverage=base.protective_put_coverage,
            tail_hedge_cost_annual=base.tail_hedge_cost_annual,
            tail_hedge_trigger=base.tail_hedge_trigger,
            tail_hedge_slope=base.tail_hedge_slope,
            collar_floor=floor,
            collar_cap=cap,
            collar_cost_annual=cost,
            covered_call_write_fraction=base.covered_call_write_fraction,
            covered_call_strike_otm=base.covered_call_strike_otm,
            covered_call_premium_annual=base.covered_call_premium_annual,
            covered_call_assignment_cost=base.covered_call_assignment_cost,
        )
    if strategy == "covered_call":
        premium = _strategy_cost_model(strategy, params)
        return HedgingConfig(
            enabled=True,
            strategy="covered_call",
            rebalance_frequency=frequency,
            protective_put_cost_annual=base.protective_put_cost_annual,
            protective_put_floor=base.protective_put_floor,
            protective_put_coverage=base.protective_put_coverage,
            tail_hedge_cost_annual=base.tail_hedge_cost_annual,
            tail_hedge_trigger=base.tail_hedge_trigger,
            tail_hedge_slope=base.tail_hedge_slope,
            collar_floor=base.collar_floor,
            collar_cap=base.collar_cap,
            collar_cost_annual=base.collar_cost_annual,
            covered_call_write_fraction=float(params["covered_call_write_fraction"]),
            covered_call_strike_otm=float(params["covered_call_strike_otm"]),
            covered_call_premium_annual=premium,
            covered_call_assignment_cost=base.covered_call_assignment_cost,
        )
    raise ValueError(f"Unsupported strategy for tuning: {strategy}")


def _suggest_strategy_params(
    trial: optuna.Trial,
    strategy: Literal["protective_put", "tail_hedge", "collar", "covered_call"],
) -> dict[str, float]:
    if strategy == "protective_put":
        return {
            "protective_put_floor": trial.suggest_float(
                "protective_put_floor", -0.14, -0.08, step=0.01
            ),
            "protective_put_coverage": float(
                trial.suggest_categorical(
                    "protective_put_coverage",
                    list(PROTECTIVE_PUT_COVERAGE_CANDIDATES),
                )
            ),
            "premium_adjustment": trial.suggest_float("premium_adjustment", 0.0, 0.015, step=0.001),
        }
    if strategy == "tail_hedge":
        return {
            "tail_hedge_trigger": trial.suggest_float(
                "tail_hedge_trigger", -0.20, -0.12, step=0.01
            ),
            "tail_hedge_slope": trial.suggest_float("tail_hedge_slope", 0.25, 0.80, step=0.05),
            "premium_adjustment": trial.suggest_float("premium_adjustment", 0.0, 0.01, step=0.001),
        }
    if strategy == "collar":
        return {
            "collar_floor": trial.suggest_float("collar_floor", -0.12, -0.06, step=0.01),
            "collar_cap": trial.suggest_float("collar_cap", 0.03, 0.08, step=0.01),
            "premium_adjustment": trial.suggest_float("premium_adjustment", 0.0, 0.008, step=0.001),
        }
    if strategy == "covered_call":
        return {
            "covered_call_write_fraction": float(
                trial.suggest_categorical(
                    "covered_call_write_fraction",
                    list(COVERED_CALL_WRITE_FRACTION_CANDIDATES),
                )
            ),
            "covered_call_strike_otm": float(
                trial.suggest_categorical(
                    "covered_call_strike_otm",
                    list(COVERED_CALL_STRIKE_OTM_CANDIDATES),
                )
            ),
            "premium_adjustment": trial.suggest_float("premium_adjustment", -0.001, 0.002, step=0.001),
        }
    raise ValueError(f"Unsupported strategy for tuning: {strategy}")


def _default_params_for_strategy(
    strategy: Literal["protective_put", "tail_hedge", "collar", "covered_call"],
    frequency: Literal["monthly", "yearly"] = SIMULATION_FREQUENCY,
) -> dict[str, float]:
    defaults = _default_strategy_config_map(frequency=frequency)
    base = defaults[strategy]
    if strategy == "protective_put":
        return {
            "protective_put_floor": float(base.protective_put_floor),
            # Keep fallback inside tuned categorical coverage candidates.
            "protective_put_coverage": float(PROTECTIVE_PUT_FALLBACK_COVERAGE),
            "premium_adjustment": 0.0,
        }
    if strategy == "tail_hedge":
        return {
            "tail_hedge_trigger": float(base.tail_hedge_trigger),
            "tail_hedge_slope": float(base.tail_hedge_slope),
            "premium_adjustment": 0.0,
        }
    if strategy == "collar":
        return {
            "collar_floor": float(base.collar_floor),
            "collar_cap": float(base.collar_cap),
            "premium_adjustment": 0.0,
        }
    if strategy == "covered_call":
        return {
            "covered_call_write_fraction": float(base.covered_call_write_fraction),
            "covered_call_strike_otm": float(base.covered_call_strike_otm),
            "premium_adjustment": 0.0,
        }
    raise ValueError(f"Unsupported strategy for defaults: {strategy}")


def _failure_first_objective_score(
    *,
    baseline_failure_rate: float,
    strategy_failure_rate: float,
    baseline_cagr: float,
    strategy_cagr: float,
    baseline_max_drawdown: float,
    strategy_max_drawdown: float,
    annual_drag: float,
    activation_rate: float,
) -> float:
    failure_reduction = baseline_failure_rate - strategy_failure_rate
    cagr_delta = strategy_cagr - baseline_cagr
    drawdown_improvement = strategy_max_drawdown - baseline_max_drawdown
    excess_cagr_delta = max(0.0, cagr_delta - OPTUNA_MAX_REASONABLE_CAGR_DELTA)
    return 100.0 * (
        0.72 * failure_reduction
        + 0.15 * drawdown_improvement
        + 0.08 * cagr_delta
        - OPTUNA_OBJECTIVE_DRAG_WEIGHT * annual_drag
        - OPTUNA_OBJECTIVE_ACTIVATION_WEIGHT * activation_rate
        - 1.00 * excess_cagr_delta
    )


def _plausibility_penalty(
    *,
    success_rate: float,
    delta_cagr: float,
    annual_drag: float,
    activation_rate: float,
) -> float:
    penalty = 0.0
    if success_rate >= OPTUNA_NEAR_PERFECT_SUCCESS_RATE and annual_drag < OPTUNA_MIN_DRAG_FOR_NEAR_PERFECT_SUCCESS:
        penalty += 8.0
    if delta_cagr > OPTUNA_MAX_REASONABLE_CAGR_DELTA:
        penalty += 100.0 * (delta_cagr - OPTUNA_MAX_REASONABLE_CAGR_DELTA)
    if activation_rate > OPTUNA_MAX_REASONABLE_ACTIVATION_RATE and annual_drag < 0.02:
        penalty += 4.0
    return penalty


def _as_float(value: float | str | None, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Expected numeric field {field_name}, got None")
    return float(value)


def _strategy_reason_against_baseline(
    *,
    strategy: str,
    delta_failure_rate: float,
    delta_cagr: float,
    annual_drag: float,
    activation_rate: float,
) -> str:
    if strategy == "none":
        return "baseline_reference"
    reason_parts: list[str] = []
    if delta_failure_rate < 0:
        reason_parts.append("lower_failure")
    elif delta_failure_rate > 0:
        reason_parts.append("higher_failure")
    if delta_cagr > 0:
        reason_parts.append("higher_cagr")
    elif delta_cagr < 0:
        reason_parts.append("cagr_drag")
    if annual_drag > 0:
        reason_parts.append("premium_cost")
    if activation_rate > 0:
        reason_parts.append("active_usage")
    return ",".join(reason_parts)


def _merge_strategy_with_baseline(
    *,
    baseline: dict[str, float | str | None],
    strategy_row: dict[str, float | str | None],
) -> dict[str, float | str | None]:
    out = dict(strategy_row)
    baseline_success = _as_float(baseline["success_rate"], "success_rate")
    baseline_failure = _as_float(baseline["failure_rate"], "failure_rate")
    baseline_cagr = _as_float(baseline["cagr"], "cagr")
    baseline_median = _as_float(baseline["median_final_balance"], "median_final_balance")
    baseline_mdd = _as_float(baseline["max_drawdown"], "max_drawdown")

    strategy_success = _as_float(strategy_row["success_rate"], "success_rate")
    strategy_failure = _as_float(strategy_row["failure_rate"], "failure_rate")
    strategy_cagr = _as_float(strategy_row["cagr"], "cagr")
    strategy_median = _as_float(strategy_row["median_final_balance"], "median_final_balance")
    strategy_mdd = _as_float(strategy_row["max_drawdown"], "max_drawdown")
    annual_drag = _as_float(
        strategy_row["avg_annual_hedging_drag_estimate"],
        "avg_annual_hedging_drag_estimate",
    )
    activation_rate = _as_float(strategy_row["activation_rate"], "activation_rate")

    out["delta_success_rate"] = strategy_success - baseline_success
    out["delta_failure_rate"] = strategy_failure - baseline_failure
    out["delta_cagr"] = strategy_cagr - baseline_cagr
    out["delta_median_final_balance"] = strategy_median - baseline_median
    out["delta_max_drawdown"] = strategy_mdd - baseline_mdd

    failure_reduction = baseline_failure - strategy_failure
    drawdown_improvement = strategy_mdd - baseline_mdd
    cagr_delta = strategy_cagr - baseline_cagr
    worth_score = 100.0 * (
        FAILURE_REDUCTION_WEIGHT * failure_reduction
        + DRAWDOWN_IMPROVEMENT_WEIGHT * drawdown_improvement
        + CAGR_WEIGHT * cagr_delta
        - HEDGE_DRAG_WEIGHT * annual_drag
    )
    out["worth_it_score"] = worth_score
    out["recommendation"] = _recommendation_for_score(worth_score)
    out["recommendation_reason"] = _strategy_reason_against_baseline(
        strategy=str(strategy_row["strategy"]),
        delta_failure_rate=strategy_failure - baseline_failure,
        delta_cagr=cagr_delta,
        annual_drag=annual_drag,
        activation_rate=activation_rate,
    )
    return out


def _evaluate_strategy_vs_baseline(
    *,
    daily_returns: np.ndarray,
    strategy_config: HedgingConfig,
    random_seed: int | None,
    n_sims: int,
    baseline_cache: dict[tuple[int | None, int, int, int, int], dict[str, float | str | None]],
    sample_mode: Literal["iid", "block_bootstrap"] = SIMULATION_SAMPLE_MODE,
    block_days: int = SIMULATION_BLOCK_DAYS,
    data_key: int = 0,
) -> dict[str, float | str | None]:
    cache_key = (
        random_seed,
        n_sims,
        1 if sample_mode == "block_bootstrap" else 0,
        block_days,
        data_key,
    )
    if cache_key not in baseline_cache:
        baseline_cache[cache_key] = summarize_simulation(
            hedging_config=HedgingConfig(
                enabled=False,
                strategy="none",
                rebalance_frequency=SIMULATION_FREQUENCY,
            ),
            daily_returns=daily_returns,
            random_seed=random_seed,
            n_sims=n_sims,
            sample_mode=sample_mode,
            block_days=block_days,
        )

    strategy_row = summarize_simulation(
        hedging_config=strategy_config,
        daily_returns=daily_returns,
        random_seed=random_seed,
        n_sims=n_sims,
        sample_mode=sample_mode,
        block_days=block_days,
    )
    return _merge_strategy_with_baseline(
        baseline=baseline_cache[cache_key],
        strategy_row=strategy_row,
    )


def _robustness_summary(
    *, rows: list[dict[str, float | str | None]], strategy: str
) -> dict[str, float | str]:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No robustness rows to summarize")
    metrics = ["failure_rate", "success_rate", "cagr", "worth_it_score"]
    out: dict[str, float | str] = {"strategy": strategy}
    for metric in metrics:
        vals = pd.to_numeric(df[metric], errors="coerce")
        out[f"{metric}_min"] = float(vals.min())
        out[f"{metric}_median"] = float(vals.median())
        out[f"{metric}_max"] = float(vals.max())
    return out


def _format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    pct_cols = [
        "success_rate",
        "failure_rate",
        "cagr",
        "max_drawdown",
        "volatility",
        "avg_annual_hedging_drag_estimate",
        "activation_rate",
        "floor_hit_rate",
        "cap_hit_rate",
        "assignment_rate",
        "delta_success_rate",
        "delta_failure_rate",
        "delta_cagr",
        "delta_max_drawdown",
    ]
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: "N/A" if pd.isna(x) else f"{float(x):.2%}"
            )
    if "protective_put_coverage" in display_df.columns:
        display_df["protective_put_coverage"] = display_df["protective_put_coverage"].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.0%}"
        )
    if "covered_call_write_fraction" in display_df.columns:
        display_df["covered_call_write_fraction"] = display_df["covered_call_write_fraction"].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.0%}"
        )
    if "covered_call_strike_otm" in display_df.columns:
        display_df["covered_call_strike_otm"] = display_df["covered_call_strike_otm"].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.0%}"
        )
    if "coverage_candidate" in display_df.columns:
        display_df["coverage_candidate"] = display_df["coverage_candidate"].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.0%}"
        )
    if "write_fraction_candidate" in display_df.columns:
        display_df["write_fraction_candidate"] = display_df["write_fraction_candidate"].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.0%}"
        )
    if "selected_strike_otm" in display_df.columns:
        display_df["selected_strike_otm"] = display_df["selected_strike_otm"].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.0%}"
        )
    if "median_final_balance" in display_df.columns:
        display_df["median_final_balance"] = display_df["median_final_balance"].map(
            lambda x: f"${float(x):,.0f}"
        )
    if "n_sims" in display_df.columns:
        display_df["n_sims"] = display_df["n_sims"].map(lambda x: f"{int(x):,}")
    int_cols = [
        "contracts_processed",
        "contract_periods_evaluated_total",
        "new_hedges_opened_total",
        "assignment_count_total",
        "entry_cap_days",
        "max_possible_new_hedges_per_year_from_cap",
    ]
    for col in int_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: "N/A" if pd.isna(x) else f"{int(float(x)):,}"
            )
    if "avg_new_hedges_per_sim_per_year" in display_df.columns:
        display_df["avg_new_hedges_per_sim_per_year"] = display_df[
            "avg_new_hedges_per_sim_per_year"
        ].map(lambda x: "N/A" if pd.isna(x) else f"{float(x):.2f}")
    if "avg_assignments_per_sim_per_year" in display_df.columns:
        display_df["avg_assignments_per_sim_per_year"] = display_df[
            "avg_assignments_per_sim_per_year"
        ].map(lambda x: "N/A" if pd.isna(x) else f"{float(x):.2f}")
    if "delta_median_final_balance" in display_df.columns:
        display_df["delta_median_final_balance"] = display_df["delta_median_final_balance"].map(
            lambda x: "N/A" if pd.isna(x) else f"${float(x):,.0f}"
        )
    if "worth_it_score" in display_df.columns:
        display_df["worth_it_score"] = display_df["worth_it_score"].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.2f}"
        )
    return display_df


def _optuna_columns(df: pd.DataFrame) -> list[str]:
    compact = [
        "strategy",
        "protective_put_coverage",
        "covered_call_write_fraction",
        "covered_call_strike_otm",
        "rank",
        "recommendation",
        "n_sims",
        "success_rate",
        "failure_rate",
        "median_final_balance",
        "cagr",
        "max_drawdown",
        "avg_annual_hedging_drag_estimate",
        "recommendation_reason",
    ]
    full = list(df.columns)
    selected = compact if OUTPUT_VIEW == "compact" else full
    return [c for c in selected if c in df.columns]


def _comparison_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    perf_full = [
        "strategy",
        "protective_put_coverage",
        "covered_call_write_fraction",
        "covered_call_strike_otm",
        "rank",
        "recommendation",
        "n_sims",
        "success_rate",
        "failure_rate",
        "median_final_balance",
        "cagr",
        "max_drawdown",
        "volatility",
        "delta_success_rate",
        "delta_failure_rate",
        "delta_cagr",
        "delta_median_final_balance",
        "delta_max_drawdown",
    ]
    perf_compact = [
        "strategy",
        "protective_put_coverage",
        "covered_call_write_fraction",
        "covered_call_strike_otm",
        "rank",
        "recommendation",
        "success_rate",
        "failure_rate",
        "median_final_balance",
        "cagr",
        "max_drawdown",
        "avg_annual_hedging_drag_estimate",
    ]
    diag_cols = [
        "strategy",
        "protective_put_coverage",
        "covered_call_write_fraction",
        "covered_call_strike_otm",
        "contract_periods_evaluated_total",
        "new_hedges_opened_total",
        "avg_new_hedges_per_sim_per_year",
        "assignment_rate",
        "assignment_count_total",
        "avg_assignments_per_sim_per_year",
        "max_possible_new_hedges_per_year_from_cap",
        "entry_cap_days",
        "reset_policy",
        "activation_rate",
        "floor_hit_rate",
        "cap_hit_rate",
        "avg_annual_hedging_drag_estimate",
        "worth_it_score",
        "recommendation_reason",
    ]
    perf_selected = perf_compact if OUTPUT_VIEW == "compact" else perf_full
    perf = [c for c in perf_selected if c in df.columns]
    diag = [c for c in diag_cols if c in df.columns]
    return perf, diag


def _best_params_per_put_coverage_from_trials(
    trials: list[optuna.trial.FrozenTrial],
) -> dict[float, dict[str, float]]:
    best_by_cov: dict[float, tuple[float, dict[str, float]]] = {}
    for trial in trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        if "protective_put_coverage" not in trial.params:
            continue
        cov = float(trial.params["protective_put_coverage"])
        params = {k: float(v) for k, v in trial.params.items()}
        current = best_by_cov.get(cov)
        if current is None or float(trial.value) > current[0]:
            best_by_cov[cov] = (float(trial.value), params)
    return {cov: params for cov, (_, params) in best_by_cov.items()}


def _best_params_per_covered_call_write_fraction_from_trials(
    trials: list[optuna.trial.FrozenTrial],
) -> dict[float, dict[str, float]]:
    best_by_write: dict[float, tuple[float, dict[str, float]]] = {}
    for trial in trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        if "covered_call_write_fraction" not in trial.params:
            continue
        write_fraction = float(trial.params["covered_call_write_fraction"])
        params = {k: float(v) for k, v in trial.params.items()}
        current = best_by_write.get(write_fraction)
        if current is None or float(trial.value) > current[0]:
            best_by_write[write_fraction] = (float(trial.value), params)
    return {write_fraction: params for write_fraction, (_, params) in best_by_write.items()}


def _format_best_param_value(key: str, value: float) -> str:
    if key == "protective_put_coverage":
        return f"{value:.0%}"
    if key in {"covered_call_write_fraction", "covered_call_strike_otm"}:
        return f"{value:.0%}"
    if "floor" in key or "trigger" in key or "cap" in key:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _best_params_display_df(best_params_by_strategy: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for strategy in sorted(best_params_by_strategy.keys()):
        params = best_params_by_strategy[strategy]
        if strategy == "protective_put":
            rows.append(
                {
                    "strategy": strategy,
                    "selected_coverage": _format_best_param_value(
                        "protective_put_coverage",
                        float(params.get("protective_put_coverage", np.nan)),
                    ),
                    "selected_floor": _format_best_param_value(
                        "protective_put_floor",
                        float(params.get("protective_put_floor", np.nan)),
                    ),
                    "selected_premium_adjustment": _format_best_param_value(
                        "premium_adjustment",
                        float(params.get("premium_adjustment", np.nan)),
                    ),
                }
            )
            continue
        if strategy == "covered_call":
            rows.append(
                {
                    "strategy": strategy,
                    "selected_coverage": _format_best_param_value(
                        "covered_call_write_fraction",
                        float(params.get("covered_call_write_fraction", np.nan)),
                    ),
                    "selected_floor": _format_best_param_value(
                        "covered_call_strike_otm",
                        float(params.get("covered_call_strike_otm", np.nan)),
                    ),
                    "selected_premium_adjustment": _format_best_param_value(
                        "premium_adjustment",
                        float(params.get("premium_adjustment", np.nan)),
                    ),
                }
            )
            continue

        summary = ", ".join(
            f"{k}={_format_best_param_value(k, float(v))}"
            for k, v in sorted(params.items())
        )
        rows.append(
            {
                "strategy": strategy,
                "selected_coverage": "N/A",
                "selected_floor": "N/A",
                "selected_premium_adjustment": summary,
            }
        )

    return pd.DataFrame(rows)


def _fallback_warning_lines(
    fallback_counts_by_strategy: dict[str, int],
    *,
    n_folds: int,
) -> list[str]:
    lines: list[str] = []
    for strategy in sorted(fallback_counts_by_strategy.keys()):
        count = int(fallback_counts_by_strategy[strategy])
        if count <= 0:
            continue
        if count >= n_folds:
            extra = ""
            if strategy == "protective_put":
                extra = (
                    f" (fallback coverage forced to "
                    f"{PROTECTIVE_PUT_FALLBACK_COVERAGE:.0%})"
                )
            lines.append(
                f"  - {strategy}: default fallback used in all folds "
                f"({count}/{n_folds}); final params are default-driven.{extra}"
            )
        else:
            lines.append(
                f"  - {strategy}: default fallback used in {count}/{n_folds} folds."
            )
    return lines


def _trial_completion_summary_lines(
    trial_outcomes_by_strategy: dict[str, dict[str, int]],
) -> list[str]:
    lines: list[str] = []
    for strategy in sorted(trial_outcomes_by_strategy.keys()):
        o = trial_outcomes_by_strategy[strategy]
        lines.append(
            "  - "
            f"{strategy}: complete={o.get('complete', 0)}, "
            f"pruned={o.get('pruned', 0)}, "
            f"failed={o.get('fail', 0)}"
        )
    return lines


def run_optuna_experiments() -> dict[str, object]:
    daily_returns = load_daily_returns_from_spx_csv()
    tuning_returns, holdout_returns = split_tuning_and_holdout_returns(
        daily_returns,
        holdout_years=OPTUNA_HOLDOUT_YEARS,
    )
    folds = build_walk_forward_folds(tuning_returns)
    tuned_strategies: list[Literal["protective_put", "tail_hedge", "collar", "covered_call"]] = [
        "protective_put",
        "tail_hedge",
        "collar",
        "covered_call",
    ]

    baseline_cache: dict[
        tuple[int | None, int, int, int, int], dict[str, float | str | None]
    ] = {}
    best_params_by_strategy: dict[str, dict[str, float]] = {}
    fallback_counts_by_strategy: dict[str, int] = {}
    trial_outcomes_by_strategy: dict[str, dict[str, int]] = {}
    best_put_params_by_coverage: dict[float, dict[str, float]] = {}
    best_covered_call_params_by_write_fraction: dict[float, dict[str, float]] = {}
    holdout_rows: list[dict[str, float | str | None]] = []
    put_coverage_rows: list[dict[str, float | str | None]] = []
    covered_call_write_rows: list[dict[str, float | str | None]] = []
    robustness_rows: list[dict[str, float | str | None]] = []
    robustness_summaries: list[dict[str, float | str]] = []
    fold_rows: list[dict[str, float | str | None]] = []
    protective_put_complete_trials: list[optuna.trial.FrozenTrial] = []
    covered_call_complete_trials: list[optuna.trial.FrozenTrial] = []
    trials_per_fold = max(1, OPTUNA_TRIAL_COUNT_PER_STRATEGY // len(folds))

    for strategy in tuned_strategies:
        fallback_counts_by_strategy[strategy] = 0
        trial_outcomes_by_strategy[strategy] = {"complete": 0, "pruned": 0, "fail": 0}
        fold_best_params: list[dict[str, float]] = []
        fold_best_scores: list[float] = []

        for fold in folds:
            study_name = f"{OPTUNA_STUDY_PREFIX}_{strategy}_fold{fold.fold_id}"
            try:
                study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE_PATH)
                print(f"Resuming Optuna study: {study_name}")
            except KeyError:
                study = optuna.create_study(
                    study_name=study_name,
                    storage=OPTUNA_STORAGE_PATH,
                    direction="maximize",
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
                        n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
                    ),
                )
                print(f"Created Optuna study: {study_name}")

            def objective(trial: optuna.Trial) -> float:
                params = _suggest_strategy_params(trial, strategy)
                cfg = _build_hedging_config_from_params(strategy, params)

                small_row = _evaluate_strategy_vs_baseline(
                    daily_returns=fold.train_returns,
                    strategy_config=cfg,
                    random_seed=SIMULATION_RANDOM_SEED,
                    n_sims=OPTUNA_N_SIMS_SMALL,
                    baseline_cache=baseline_cache,
                    sample_mode="iid",
                    block_days=SIMULATION_BLOCK_DAYS,
                    data_key=10_000 + fold.fold_id,
                )
                score_small = _failure_first_objective_score(
                    baseline_failure_rate=0.0,
                    strategy_failure_rate=_as_float(small_row["delta_failure_rate"], "delta_failure_rate"),
                    baseline_cagr=0.0,
                    strategy_cagr=_as_float(small_row["delta_cagr"], "delta_cagr"),
                    baseline_max_drawdown=0.0,
                    strategy_max_drawdown=_as_float(small_row["delta_max_drawdown"], "delta_max_drawdown"),
                    annual_drag=_as_float(
                        small_row["avg_annual_hedging_drag_estimate"],
                        "avg_annual_hedging_drag_estimate",
                    ),
                    activation_rate=_as_float(small_row["activation_rate"], "activation_rate"),
                )
                small_penalty = _plausibility_penalty(
                    success_rate=_as_float(small_row["success_rate"], "success_rate"),
                    delta_cagr=_as_float(small_row["delta_cagr"], "delta_cagr"),
                    annual_drag=_as_float(
                        small_row["avg_annual_hedging_drag_estimate"],
                        "avg_annual_hedging_drag_estimate",
                    ),
                    activation_rate=_as_float(small_row["activation_rate"], "activation_rate"),
                )
                score_small -= small_penalty
                trial.report(score_small, step=0)
                if OPTUNA_ENABLE_PRUNING and trial.should_prune():
                    raise optuna.TrialPruned()
                if (
                    OPTUNA_ENABLE_PRUNING
                    and _as_float(small_row["delta_failure_rate"], "delta_failure_rate")
                    > OPTUNA_EARLY_PRUNE_FAILURE_MARGIN
                ):
                    raise optuna.TrialPruned()
                if OPTUNA_ENABLE_PRUNING and small_penalty >= OPTUNA_PRUNE_PENALTY_THRESHOLD_SMALL:
                    raise optuna.TrialPruned()

                large_row = _evaluate_strategy_vs_baseline(
                    daily_returns=fold.train_returns,
                    strategy_config=cfg,
                    random_seed=SIMULATION_RANDOM_SEED,
                    n_sims=OPTUNA_N_SIMS_LARGE,
                    baseline_cache=baseline_cache,
                    sample_mode="iid",
                    block_days=SIMULATION_BLOCK_DAYS,
                    data_key=10_000 + fold.fold_id,
                )
                score_large = _failure_first_objective_score(
                    baseline_failure_rate=0.0,
                    strategy_failure_rate=_as_float(large_row["delta_failure_rate"], "delta_failure_rate"),
                    baseline_cagr=0.0,
                    strategy_cagr=_as_float(large_row["delta_cagr"], "delta_cagr"),
                    baseline_max_drawdown=0.0,
                    strategy_max_drawdown=_as_float(large_row["delta_max_drawdown"], "delta_max_drawdown"),
                    annual_drag=_as_float(
                        large_row["avg_annual_hedging_drag_estimate"],
                        "avg_annual_hedging_drag_estimate",
                    ),
                    activation_rate=_as_float(large_row["activation_rate"], "activation_rate"),
                )
                large_penalty = _plausibility_penalty(
                    success_rate=_as_float(large_row["success_rate"], "success_rate"),
                    delta_cagr=_as_float(large_row["delta_cagr"], "delta_cagr"),
                    annual_drag=_as_float(
                        large_row["avg_annual_hedging_drag_estimate"],
                        "avg_annual_hedging_drag_estimate",
                    ),
                    activation_rate=_as_float(large_row["activation_rate"], "activation_rate"),
                )
                score_large -= large_penalty
                trial.report(score_large, step=1)
                if OPTUNA_ENABLE_PRUNING and trial.should_prune():
                    raise optuna.TrialPruned()
                if OPTUNA_ENABLE_PRUNING and large_penalty >= OPTUNA_PRUNE_PENALTY_THRESHOLD_LARGE:
                    raise optuna.TrialPruned()
                trial.set_user_attr("failure_rate", _as_float(large_row["failure_rate"], "failure_rate"))
                trial.set_user_attr("cagr", _as_float(large_row["cagr"], "cagr"))
                trial.set_user_attr("plausibility_penalty", float(large_penalty))
                return score_large

            complete_before = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            )
            pruned_before = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            )
            fail_before = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
            )
            study.optimize(objective, n_trials=trials_per_fold, n_jobs=1)
            complete_after = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            )
            pruned_after = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            )
            fail_after = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
            )
            trial_outcomes_by_strategy[strategy]["complete"] += max(0, complete_after - complete_before)
            trial_outcomes_by_strategy[strategy]["pruned"] += max(0, pruned_after - pruned_before)
            trial_outcomes_by_strategy[strategy]["fail"] += max(0, fail_after - fail_before)
            complete_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not complete_trials:
                best_params = _default_params_for_strategy(strategy)
                fallback_counts_by_strategy[strategy] += 1
            else:
                best_params = {
                    k: float(v) for k, v in study.best_trial.params.items()
                }
                if strategy == "protective_put":
                    protective_put_complete_trials.extend(complete_trials)
                if strategy == "covered_call":
                    covered_call_complete_trials.extend(complete_trials)
            fold_best_params.append(best_params)
            if not complete_trials:
                fold_best_scores.append(float("-inf"))
            else:
                fold_best_scores.append(float(study.best_value))

            best_cfg = _build_hedging_config_from_params(strategy, best_params)
            fold_eval = _evaluate_strategy_vs_baseline(
                daily_returns=fold.test_returns,
                strategy_config=best_cfg,
                random_seed=SIMULATION_RANDOM_SEED,
                n_sims=WF_VALIDATION_N_SIMS,
                baseline_cache=baseline_cache,
                sample_mode="block_bootstrap",
                block_days=SIMULATION_BLOCK_DAYS,
                data_key=20_000 + fold.fold_id,
            )
            fold_eval["strategy"] = strategy
            fold_eval["fold_id"] = fold.fold_id
            fold_rows.append(fold_eval)

        if not fold_best_scores:
            raise ValueError(f"No fold results for strategy {strategy}")
        best_fold_index = int(np.argmax(np.asarray(fold_best_scores)))
        selected_params = fold_best_params[best_fold_index]
        best_params_by_strategy[strategy] = selected_params
        best_cfg = _build_hedging_config_from_params(strategy, selected_params)

        holdout_row = _evaluate_strategy_vs_baseline(
            daily_returns=holdout_returns,
            strategy_config=best_cfg,
            random_seed=SIMULATION_RANDOM_SEED,
            n_sims=N_SIMS,
            baseline_cache=baseline_cache,
            sample_mode="block_bootstrap",
            block_days=SIMULATION_BLOCK_DAYS,
            data_key=40_000 + tuned_strategies.index(strategy),
        )
        holdout_rows.append(holdout_row)

        per_seed_rows: list[dict[str, float | str | None]] = []
        for seed in ROBUSTNESS_SEEDS:
            rng = np.random.default_rng(seed)
            for draw in range(BOOTSTRAP_DRAWS):
                boot_returns = block_bootstrap_returns(
                    holdout_returns,
                    block_days=BOOTSTRAP_BLOCK_DAYS,
                    rng=rng,
                )
                row = _evaluate_strategy_vs_baseline(
                    daily_returns=boot_returns,
                    strategy_config=best_cfg,
                    random_seed=seed + draw,
                    n_sims=BOOTSTRAP_N_SIMS,
                    baseline_cache=baseline_cache,
                    sample_mode="iid",
                    block_days=SIMULATION_BLOCK_DAYS,
                    data_key=30_000 + (seed * 100) + draw,
                )
                row["seed"] = int(seed)
                row["draw"] = int(draw)
                per_seed_rows.append(row)
                robustness_rows.append(row)
        robustness_summaries.append(_robustness_summary(rows=per_seed_rows, strategy=strategy))

    best_put_params_by_coverage = _best_params_per_put_coverage_from_trials(
        protective_put_complete_trials
    )
    for cov in PROTECTIVE_PUT_COVERAGE_CANDIDATES:
        params = best_put_params_by_coverage.get(float(cov))
        source = "best_trial"
        if params is None:
            source = "default_fallback"
            params = _default_params_for_strategy("protective_put")
            params["protective_put_coverage"] = float(cov)
        cfg = _build_hedging_config_from_params("protective_put", params)
        row = _evaluate_strategy_vs_baseline(
            daily_returns=holdout_returns,
            strategy_config=cfg,
            random_seed=SIMULATION_RANDOM_SEED,
            n_sims=N_SIMS,
            baseline_cache=baseline_cache,
            sample_mode="block_bootstrap",
            block_days=SIMULATION_BLOCK_DAYS,
            data_key=50_000 + int(round(float(cov) * 100)),
        )
        row["coverage_candidate"] = float(cov)
        row["selection_source"] = source
        put_coverage_rows.append(row)

    best_covered_call_params_by_write_fraction = (
        _best_params_per_covered_call_write_fraction_from_trials(covered_call_complete_trials)
    )
    for write_fraction in COVERED_CALL_WRITE_FRACTION_CANDIDATES:
        params = best_covered_call_params_by_write_fraction.get(float(write_fraction))
        source = "best_trial"
        if params is None:
            source = "default_fallback"
            params = _default_params_for_strategy("covered_call")
            params["covered_call_write_fraction"] = float(write_fraction)
        cfg = _build_hedging_config_from_params("covered_call", params)
        row = _evaluate_strategy_vs_baseline(
            daily_returns=holdout_returns,
            strategy_config=cfg,
            random_seed=SIMULATION_RANDOM_SEED,
            n_sims=N_SIMS,
            baseline_cache=baseline_cache,
            sample_mode="block_bootstrap",
            block_days=SIMULATION_BLOCK_DAYS,
            data_key=60_000 + int(round(float(write_fraction) * 100)),
        )
        row["write_fraction_candidate"] = float(write_fraction)
        row["selected_strike_otm"] = float(params["covered_call_strike_otm"])
        row["selection_source"] = source
        covered_call_write_rows.append(row)

    baseline_validation = summarize_simulation(
        hedging_config=HedgingConfig(
            enabled=False,
            strategy="none",
            rebalance_frequency=SIMULATION_FREQUENCY,
        ),
        daily_returns=holdout_returns,
        random_seed=SIMULATION_RANDOM_SEED,
        n_sims=N_SIMS,
        sample_mode="block_bootstrap",
        block_days=SIMULATION_BLOCK_DAYS,
    )

    final_df = add_decision_analytics(pd.DataFrame([baseline_validation, *holdout_rows]))
    final_df = final_df.sort_values("rank")
    put_coverage_df = pd.DataFrame(put_coverage_rows).sort_values(
        "coverage_candidate", ascending=False
    )
    covered_call_write_df = pd.DataFrame(covered_call_write_rows).sort_values(
        "write_fraction_candidate", ascending=False
    )
    robustness_df = pd.DataFrame(robustness_summaries).sort_values("strategy")
    display_final_df = _format_display_df(final_df)
    display_put_coverage_df = _format_display_df(put_coverage_df)
    display_covered_call_write_df = _format_display_df(covered_call_write_df)
    display_robustness_df = _format_display_df(robustness_df)

    print("\n=== Optuna Best-Config Comparison ===")
    print(
        f"Data source: {SPX_DAILY_FILE_PATH} (daily returns)\n"
        f"Option reset horizon: every {OPTION_DAYS} trading days ({HEDGE_RESET_POLICY})\n"
        f"Entry cap: at most 1 new hedge every {HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES} trading days\n"
        f"Output view: {OUTPUT_VIEW}\n"
        f"Selection data: walk-forward tuning folds | Final ranking data: holdout last {OPTUNA_HOLDOUT_YEARS} years"
    )
    print(display_final_df[_optuna_columns(display_final_df)].to_string(index=False))
    coverage_candidates = ", ".join(
        f"{c:.0%}" for c in PROTECTIVE_PUT_COVERAGE_CANDIDATES
    )
    print(
        "\nProtective Put Coverage Candidates Tested: "
        f"{coverage_candidates}"
    )
    print("Selected Best Parameters:")
    print(_best_params_display_df(best_params_by_strategy).to_string(index=False))
    completion_lines = _trial_completion_summary_lines(trial_outcomes_by_strategy)
    if completion_lines:
        print("\nTrial Completion Summary (new trials this run):")
        for line in completion_lines:
            print(line)
    warning_lines = _fallback_warning_lines(
        fallback_counts_by_strategy,
        n_folds=len(folds),
    )
    if warning_lines:
        print("\nTuning Warnings:")
        for line in warning_lines:
            print(line)
    print("\n=== Protective Put Coverage Impact (Best per Candidate) ===")
    put_cov_cols = [
        "coverage_candidate",
        "selection_source",
        "success_rate",
        "failure_rate",
        "delta_success_rate",
        "delta_failure_rate",
        "cagr",
        "delta_cagr",
        "median_final_balance",
        "delta_median_final_balance",
        "max_drawdown",
        "avg_annual_hedging_drag_estimate",
        "recommendation",
    ]
    available_put_cov_cols = [c for c in put_cov_cols if c in display_put_coverage_df.columns]
    print(display_put_coverage_df[available_put_cov_cols].to_string(index=False))
    write_candidates = ", ".join(
        f"{w:.0%}" for w in COVERED_CALL_WRITE_FRACTION_CANDIDATES
    )
    strike_candidates = ", ".join(
        f"{s:.0%}" for s in COVERED_CALL_STRIKE_OTM_CANDIDATES
    )
    print(
        "\nCovered-Call Candidates Tested: "
        f"write={write_candidates} | strike_otm={strike_candidates}"
    )
    print("\n=== Covered-Call Impact (Best per Write Fraction) ===")
    cc_cols = [
        "write_fraction_candidate",
        "selected_strike_otm",
        "selection_source",
        "success_rate",
        "failure_rate",
        "delta_success_rate",
        "delta_failure_rate",
        "cagr",
        "delta_cagr",
        "median_final_balance",
        "delta_median_final_balance",
        "assignment_rate",
        "assignment_count_total",
        "avg_assignments_per_sim_per_year",
        "avg_annual_hedging_drag_estimate",
        "recommendation",
    ]
    available_cc_cols = [c for c in cc_cols if c in display_covered_call_write_df.columns]
    print(display_covered_call_write_df[available_cc_cols].to_string(index=False))
    print("\n=== Robustness (Multi-Seed) ===")
    print(display_robustness_df.to_string(index=False))
    if OUTPUT_VIEW == "full" or SHOW_GLOSSARY_IN_COMPACT:
        print_column_glossary(display_final_df.columns.to_list())

    payload = {
        "study_prefix": OPTUNA_STUDY_PREFIX,
        "storage_path": OPTUNA_STORAGE_PATH,
        "trial_count_per_strategy": OPTUNA_TRIAL_COUNT_PER_STRATEGY,
        "holdout_years": OPTUNA_HOLDOUT_YEARS,
        "holdout_days": int(len(holdout_returns)),
        "tuning_days": int(len(tuning_returns)),
        "best_params_by_strategy": best_params_by_strategy,
        "fallback_counts_by_strategy": fallback_counts_by_strategy,
        "trial_outcomes_by_strategy": trial_outcomes_by_strategy,
        "best_params_protective_put_by_coverage": {
            f"{float(cov):.2f}": params for cov, params in sorted(best_put_params_by_coverage.items())
        },
        "best_params_covered_call_by_write_fraction": {
            f"{float(w):.2f}": params
            for w, params in sorted(best_covered_call_params_by_write_fraction.items())
        },
        "walk_forward_folds": len(folds),
        "walk_forward_rows": pd.DataFrame(fold_rows).to_dict(orient="records"),
        "final_validation": final_df.to_dict(orient="records"),
        "final_holdout_validation": final_df.to_dict(orient="records"),
        "protective_put_coverage_impact": put_coverage_df.to_dict(orient="records"),
        "covered_call_write_fraction_impact": covered_call_write_df.to_dict(orient="records"),
        "robustness_summary": robustness_df.to_dict(orient="records"),
        "robustness_rows": pd.DataFrame(robustness_rows).to_dict(orient="records"),
    }
    os.makedirs(os.path.dirname(RESULTS_OPTUNA_PATH) or ".", exist_ok=True)
    with open(RESULTS_OPTUNA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved Optuna experiment summary: {RESULTS_OPTUNA_PATH}")
    return payload


def compare_hedging_strategies(
    frequency: Literal["monthly", "yearly"] = SIMULATION_FREQUENCY,
) -> pd.DataFrame:
    daily_returns = load_daily_returns_from_spx_csv()
    rows: list[dict[str, float | str | None]] = []
    for config in build_strategy_configs(frequency=frequency):
        rows.append(
            summarize_simulation(
                hedging_config=config,
                daily_returns=daily_returns,
                random_seed=SIMULATION_RANDOM_SEED,
            )
        )
    return add_decision_analytics(pd.DataFrame(rows))


def print_comparison(df: pd.DataFrame) -> None:
    print("\n=== Hedge Strategy Comparison ===")
    print(
        "Runs in order: baseline (none), protective_put, tail_hedge, collar, covered_call\n"
        f"Data source: {SPX_DAILY_FILE_PATH} (daily returns)\n"
        f"Option reset horizon: every {OPTION_DAYS} trading days ({HEDGE_RESET_POLICY})\n"
        f"Entry cap: at most 1 new hedge every {HEDGE_MIN_DAYS_BETWEEN_NEW_ENTRIES} trading days\n"
        f"Output view: {OUTPUT_VIEW}\n"
        f"Withdrawal positive year: ${WITHDRAWAL_POSITIVE_YEAR:,.0f} | "
        f"negative year: ${WITHDRAWAL_NEGATIVE_YEAR:,.0f}"
    )
    display_df = _format_display_df(df)
    available_perf, available_diag = _comparison_columns(display_df)

    print("\nPerformance vs Baseline:")
    print(display_df[available_perf].to_string(index=False))
    if OUTPUT_VIEW == "full":
        print("\nEffort/Risk Diagnostics:")
        print(display_df[available_diag].to_string(index=False))
    if OUTPUT_VIEW == "full" or SHOW_GLOSSARY_IN_COMPACT:
        print_column_glossary(display_df.columns.to_list())


def _column_glossary_lines(columns: list[str]) -> list[str]:
    glossary: dict[str, tuple[str, str]] = {
        "strategy": ("Strategy label", "n/a"),
        "protective_put_coverage": (
            "Protective-put downside coverage fraction below floor",
            "higher is stronger protection",
        ),
        "covered_call_write_fraction": (
            "Fraction of equity sleeve overlaid with covered calls",
            "context",
        ),
        "covered_call_strike_otm": (
            "Covered-call strike offset above spot at contract start",
            "context",
        ),
        "coverage_candidate": ("Candidate protective-put coverage bucket", "context"),
        "write_fraction_candidate": ("Candidate covered-call write fraction bucket", "context"),
        "selected_strike_otm": ("Selected covered-call strike offset for candidate row", "context"),
        "selection_source": ("How candidate params were selected", "context"),
        "rank": ("Rank by worth_it_score (1 is best)", "lower is better"),
        "recommendation": ("Final decision label", "worth_it > borderline > not_worth_it"),
        "n_sims": ("Number of Monte Carlo paths", "higher is more stable"),
        "success_rate": ("Share of paths ending > 0", "higher is better"),
        "failure_rate": ("Share of paths ending <= 0", "lower is better"),
        "median_final_balance": ("Median ending portfolio value", "higher is better"),
        "cagr": ("Median-path annualized growth rate", "higher is better"),
        "max_drawdown": ("Worst peak-to-trough path drawdown", "less negative is better"),
        "volatility": ("Annualized return variability", "lower usually safer"),
        "delta_success_rate": ("Strategy minus baseline success_rate", "higher is better"),
        "delta_failure_rate": ("Strategy minus baseline failure_rate", "lower is better"),
        "delta_cagr": ("Strategy minus baseline CAGR", "higher is better"),
        "delta_median_final_balance": (
            "Strategy minus baseline median_final_balance",
            "higher is better",
        ),
        "delta_max_drawdown": (
            "Strategy minus baseline max_drawdown",
            "higher is better",
        ),
        "contracts_processed": ("Alias of contract_periods_evaluated_total", "context"),
        "contract_periods_evaluated_total": ("Total 21-day periods evaluated", "context"),
        "new_hedges_opened_total": ("Total new hedge entries actually opened", "context"),
        "avg_new_hedges_per_sim_per_year": (
            "Average hedge entries per simulation-year",
            "context",
        ),
        "max_possible_new_hedges_per_year_from_cap": (
            "Entry-cap-implied max hedge entries per year",
            "context",
        ),
        "entry_cap_days": ("Minimum days between new hedge entries", "context"),
        "reset_policy": ("Hedge lifecycle reset policy", "context"),
        "activation_rate": ("Share of contracts where hedge condition activated", "context"),
        "floor_hit_rate": ("Share of contracts hitting downside floor/trigger", "context"),
        "cap_hit_rate": ("Share of contracts hitting collar upside cap", "context"),
        "assignment_rate": ("Share of contracts where covered calls are assigned", "context"),
        "assignment_count_total": ("Total covered-call assignment events", "context"),
        "avg_assignments_per_sim_per_year": (
            "Average covered-call assignments per simulation-year",
            "context",
        ),
        "avg_annual_hedging_drag_estimate": (
            "Modeled annual hedge cost drag",
            "lower is better",
        ),
        "worth_it_score": ("Composite decision score", "higher is better"),
        "recommendation_reason": ("Short reason tags behind recommendation", "context"),
    }
    lines = ["\nColumn Glossary:"]
    for column in columns:
        if column in glossary:
            meaning, direction = glossary[column]
            lines.append(f"  - {column}: {meaning} ({direction})")
    return lines


def print_column_glossary(columns: list[str]) -> None:
    for line in _column_glossary_lines(columns):
        print(line)


if __name__ == "__main__":
    start_time = time.perf_counter()
    if RUN_MODE == "optuna":
        run_optuna_experiments()
    else:
        comparison_df = compare_hedging_strategies()
        print_comparison(comparison_df)
    elapsed = time.perf_counter() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")
