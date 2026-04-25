import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Literal, Optional
import pandas as pd
import numpy as np
import math
from random_utils import (
    generate_block_bootstrap_indices,
    generate_constrained_indices,
)

SamplingMode = Literal["random", "constrained", "block_bootstrap"]
BondReturnMode = Literal["fixed", "historical"]


def wilson_score_interval(
    successes: int, trials: int, z: float = 1.96
) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion (more stable than normal
    approximation near 0 or 1).
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    if successes < 0 or successes > trials:
        raise ValueError(
            f"successes must be in [0, trials], got successes={successes}, trials={trials}"
        )
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denom
    inner = p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    half_width = z * math.sqrt(max(inner, 0.0)) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def binomial_proportion_standard_error(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(math.sqrt(max(p * (1.0 - p) / n, 0.0)))


class SimulationData:
    def __init__(
        self,
        initial_balance,
        withdrawal,
        withdrawal_negative_year,
        n_years,
        n_sims,
        final_balances,
        trajectories,
        total_years,
        returns_by_year,
        *,
        go_back_year: int = 0,
        reserve_floor: Optional[float] = None,
        sampling_mode: Optional[str] = None,
        block_bootstrap_size: Optional[int] = None,
        bond_return_mode: Optional[str] = None,
        inflation_rate: Optional[float] = None,
        bond_rate: Optional[float] = None,
        annual_expense_ratio: float = 0.0,
        overlapping_windows_note: Optional[str] = None,
    ):
        self.initial_balance = initial_balance
        self.withdrawal = withdrawal
        self.withdrawal_negative_year = withdrawal_negative_year
        self.n_years = n_years
        self.n_sims = n_sims
        self.final_balances = final_balances
        self.trajectories = trajectories
        self.total_years = total_years
        self.returns_by_year = returns_by_year
        self.go_back_year = go_back_year
        self.reserve_floor = reserve_floor
        self.sampling_mode = sampling_mode
        self.block_bootstrap_size = block_bootstrap_size
        self.bond_return_mode = bond_return_mode
        self.inflation_rate = inflation_rate
        self.bond_rate = bond_rate
        self.annual_expense_ratio = annual_expense_ratio
        self.overlapping_windows_note = overlapping_windows_note

        successes_arr = final_balances > 0
        self.success_count = int(np.sum(successes_arr))
        self.probability_of_success = float(np.mean(successes_arr))
        self.probability_of_success_se = binomial_proportion_standard_error(
            self.probability_of_success, n_sims
        )
        self.wilson_success_95 = wilson_score_interval(
            self.success_count, n_sims, z=1.96
        )
        self.wilson_success_99 = wilson_score_interval(
            self.success_count, n_sims, z=2.576
        )

        if reserve_floor is not None and reserve_floor < 0:
            raise ValueError(f"reserve_floor must be >= 0, got {reserve_floor}")
        self.probability_below_reserve_floor: Optional[float] = (
            float(np.mean(final_balances < float(reserve_floor)))
            if reserve_floor is not None
            else None
        )

        self.std_final = np.std(
            final_balances, ddof=1
        )  # ddof=1 for sample standard deviation
        self.std_error = self.std_final / np.sqrt(n_sims)

    def print_stats(self):
        prob_success = self.probability_of_success
        mean_final = np.mean(self.final_balances)
        median_final = np.median(self.final_balances)
        percentile_list = [1, 25, 50, 75, 99]
        percentiles = np.percentile(self.final_balances, percentile_list)
        std_final = self.std_final

        z95 = 1.96  # 95% confidence interval
        z99 = 2.576  # 99% confidence interval
        n = len(self.final_balances)
        ci_lower_95 = mean_final - z95 * (std_final / np.sqrt(n))
        ci_upper_95 = mean_final + z95 * (std_final / np.sqrt(n))
        ci_lower_99 = mean_final - z99 * (std_final / np.sqrt(n))
        ci_upper_99 = mean_final + z99 * (std_final / np.sqrt(n))

        print(
            f"\nProbability portfolio survives {self.n_years} years: {prob_success:.2%} "
            f"if withdrawing ${self.withdrawal:,.0f} per year and in negative years "
            f"${self.withdrawal_negative_year:,.0f} per year."
        )
        print(
            f"  Success rate SE (binomial): {self.probability_of_success_se:.4%} "
            f"(n={self.n_sims:,}, successes={self.success_count:,})"
        )
        w95_lo, w95_hi = self.wilson_success_95
        w99_lo, w99_hi = self.wilson_success_99
        print(
            f"  Wilson 95% interval for P(success): {w95_lo:.2%} – {w95_hi:.2%} "
            f"(not the same as mean-balance CIs below)"
        )
        print(
            f"  Wilson 99% interval for P(success): {w99_lo:.2%} – {w99_hi:.2%}"
        )

        # Separate the three categories
        underflow = self.final_balances[self.final_balances <= 0]
        overflow = self.final_balances[self.final_balances > 0]
        print(f"{len(underflow):,} simulations ended ≤ $0.")
        print(f"{len(overflow):,} simulations ended > $0.")

        print(f"\nMedian ending balance: ${median_final:,.0f}")
        print(
            "th, ".join([str(p) for p in percentile_list]) + "th percentile outcomes:",
            [f"${p:,.0f}" for p in percentiles],
        )
        print(f"Standard deviation of ending balances: ${std_final:,.0f}")
        print(f"Mean ending balance: ${mean_final:,.0f}")
        print(
            "Standard error of mean ending balance: "
            f"${self.std_error:,.0f} (this is NOT the SE of P(success))"
        )
        if mean_final != 0.0:
            print(f"Standard error / mean (ending balance): {self.std_error / mean_final:.3%}")
        else:
            print("Standard error / mean (ending balance): N/A (mean is zero)")
        print(
            f"95% CI for mean ending balance: ${ci_lower_95:,.0f} – ${ci_upper_95:,.0f}"
        )
        print(
            f"99% CI for mean ending balance: ${ci_lower_99:,.0f} – ${ci_upper_99:,.0f}"
        )

        self.print_model_risk_summary()

        if self.trajectories is not None:
            np.set_printoptions(suppress=True, precision=2, linewidth=150)

            # Year-over-year *balance* change (withdrawals + returns), not market sign.
            yearly_change = self.trajectories[:, 1:] - self.trajectories[:, :-1]
            balance_declined = yearly_change < 0
            window_size = 5
            windows = np.lib.stride_tricks.sliding_window_view(
                balance_declined, window_size, axis=1
            )

            consecutive_balance_down = np.all(windows, axis=2)

            sim_balance_down_streak = np.any(consecutive_balance_down, axis=1)

            print(
                f"{np.sum(sim_balance_down_streak)} simulations "
                f"({np.sum(sim_balance_down_streak) / len(sim_balance_down_streak):.1%}) "
                f"had {window_size} or more consecutive years where ending balance "
                f"fell below the prior year (not the same as 'negative market years')."
            )
            frac_decline = np.sum(balance_declined, axis=1) / balance_declined.shape[1]

            sim_more_than_50pct_decline = frac_decline > 0.5

            print(
                f"{np.sum(sim_more_than_50pct_decline)} simulations "
                f"({np.sum(sim_more_than_50pct_decline) / len(sim_more_than_50pct_decline):.1%}) "
                f"had balance decline in more than 50% of years."
            )

            self._print_depletion_timing_summary()

    def print_model_risk_summary(self) -> None:
        """Explicit assumptions and interpretation caveats."""
        print("\n--- Model risk / assumptions ---")
        if self.go_back_year and self.go_back_year > 0:
            print(
                f"  WARNING: go_back_year={self.go_back_year} — early years reset balance "
                f"to initial when below initial (external bailout / 'go back' rule)."
            )
        else:
            print("  go_back_year: 0 (no early reset-to-initial bailout)")
        if self.overlapping_windows_note:
            print(f"  Historical windows: {self.overlapping_windows_note}")
        if self.sampling_mode is not None:
            line = f"  sampling_mode: {self.sampling_mode}"
            if self.block_bootstrap_size is not None and self.sampling_mode == "block_bootstrap":
                line += f" (block_bootstrap_size={self.block_bootstrap_size})"
            print(line)
        if self.bond_return_mode is not None:
            print(f"  bond_return_mode: {self.bond_return_mode}")
        if self.bond_return_mode == "fixed" and self.bond_rate is not None:
            print(f"  bond_rate (nominal, fixed): {self.bond_rate:.2%}")
        elif self.bond_return_mode == "historical":
            print("  bond: nominal returns co-sampled with equities from Shiller history")
        if self.inflation_rate is not None:
            print(f"  inflation_rate (deterministic): {self.inflation_rate:.2%}")
        print(f"  annual_expense_ratio (drag on return): {self.annual_expense_ratio:.2%}")
        print(
            "  Taxes: not modeled — withdrawals are nominal; net after-tax spending "
            "should be reflected in withdrawal inputs if needed."
        )
        if self.reserve_floor is not None and self.probability_below_reserve_floor is not None:
            print(
                f"  P(ending balance < reserve floor ${self.reserve_floor:,.0f}): "
                f"{self.probability_below_reserve_floor:.2%}"
            )
        print(
            "  Timestep: annual; no intra-year path; US Shiller-based returns; "
            "see run_simulation_mp docstring for full list."
        )

    def _print_depletion_timing_summary(self) -> None:
        """If trajectories exist, summarize first year balance hits zero (if ever)."""
        if self.trajectories is None:
            return
        traj = self.trajectories
        # Years 1..n_years map to traj columns 1..n_years (column 0 is start).
        hit = traj <= 0
        any_hit = np.any(hit, axis=1)
        if not np.any(any_hit):
            print(
                "\nDepletion timing: no simulated paths reached ≤ $0 within the horizon."
            )
            return
        first_idx = np.argmax(hit, axis=1)
        # Only count sims that actually hit
        first_idx = np.where(any_hit, first_idx, -1)
        depletion_years = first_idx[first_idx >= 0]
        print(
            f"\nDepletion timing (among {np.sum(any_hit):,} paths that hit ≤ $0): "
            f"median first year index (0=start) = {float(np.median(depletion_years)):.1f}, "
            f"25th–75th pct years = {np.percentile(depletion_years, [25, 75])!s}"
        )


@dataclass(frozen=True)
class HedgingConfig:
    """
    Return-shaping hedge approximation for retirement simulations.

    This is NOT options pricing. No contracts, Greeks, implied volatility,
    rolling, execution, slippage, or broker mechanics are modeled.
    """

    enabled: bool
    strategy: Literal["none", "protective_put", "tail_hedge", "collar", "covered_call"]
    rebalance_frequency: Literal["monthly", "yearly"]
    apply_to_equity_only: bool = True

    # Protective put abstraction
    protective_put_cost_annual: float = 0.015
    protective_put_floor: float = -0.15
    # Fraction of loss below the floor offset by hedge payoff (1.0 = full floor clamp).
    protective_put_coverage: float = 1.0

    # Tail hedge abstraction
    tail_hedge_cost_annual: float = 0.005
    tail_hedge_trigger: float = -0.20
    tail_hedge_slope: float = 0.75

    # Collar abstraction
    collar_floor: float = -0.10
    collar_cap: float = 0.10
    collar_cost_annual: float = 0.0

    # Covered-call abstraction
    covered_call_write_fraction: float = 0.05
    covered_call_strike_otm: float = 0.05
    covered_call_premium_annual: float = 0.006
    covered_call_assignment_cost: float = 0.002

    # Approximate intra-period re-hedging. Example: yearly data with 12 sub-steps
    # simulates periodic hedge refreshes inside the year.
    rehedge_substeps_per_period: int = 1


def periods_per_year_from_frequency(
    frequency: Literal["monthly", "yearly"],
) -> int:
    if frequency == "monthly":
        return 12
    return 1


def annual_hedging_drag_estimate(hedging_config: HedgingConfig) -> float:
    if not hedging_config.enabled or hedging_config.strategy == "none":
        return 0.0
    if hedging_config.strategy == "protective_put":
        return hedging_config.protective_put_cost_annual
    if hedging_config.strategy == "tail_hedge":
        return hedging_config.tail_hedge_cost_annual
    if hedging_config.strategy == "collar":
        return hedging_config.collar_cost_annual
    if hedging_config.strategy == "covered_call":
        # Premium income is negative drag.
        return -hedging_config.covered_call_premium_annual
    raise ValueError(f"Unsupported hedging strategy: {hedging_config.strategy}")


def apply_hedge_to_equity_return(
    equity_return: float,
    hedging_config: HedgingConfig,
    periods_per_year: int,
) -> float:
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    if hedging_config.rehedge_substeps_per_period <= 0:
        raise ValueError(
            "rehedge_substeps_per_period must be positive, "
            f"got {hedging_config.rehedge_substeps_per_period}"
        )
    if not hedging_config.enabled or hedging_config.strategy == "none":
        return equity_return

    substeps = hedging_config.rehedge_substeps_per_period
    cost_periods_per_year = periods_per_year * substeps
    if equity_return <= -1.0:
        raw_subperiod_return = -1.0
    else:
        raw_subperiod_return = (1.0 + equity_return) ** (1.0 / substeps) - 1.0

    if hedging_config.strategy == "protective_put":
        period_cost = hedging_config.protective_put_cost_annual / cost_periods_per_year
        floor_sub = (1.0 + hedging_config.protective_put_floor) ** (1.0 / substeps) - 1.0
        coverage = float(np.clip(hedging_config.protective_put_coverage, 0.0, 1.0))
        hedged_sub = raw_subperiod_return - period_cost
        if raw_subperiod_return < floor_sub:
            hedged_sub += coverage * (floor_sub - raw_subperiod_return)
        return (1.0 + hedged_sub) ** substeps - 1.0

    if hedging_config.strategy == "tail_hedge":
        period_cost = hedging_config.tail_hedge_cost_annual / cost_periods_per_year
        trigger_sub = (1.0 + hedging_config.tail_hedge_trigger) ** (1.0 / substeps) - 1.0
        if raw_subperiod_return >= trigger_sub:
            hedged_sub = raw_subperiod_return - period_cost
        else:
            excess_drawdown = trigger_sub - raw_subperiod_return
            protection = hedging_config.tail_hedge_slope * excess_drawdown
            hedged_sub = raw_subperiod_return - period_cost + protection
        return (1.0 + hedged_sub) ** substeps - 1.0

    if hedging_config.strategy == "collar":
        period_cost = hedging_config.collar_cost_annual / cost_periods_per_year
        floor_sub = (1.0 + hedging_config.collar_floor) ** (1.0 / substeps) - 1.0
        cap_sub = (1.0 + hedging_config.collar_cap) ** (1.0 / substeps) - 1.0
        bounded = min(
            cap_sub,
            max(raw_subperiod_return, floor_sub),
        )
        hedged_sub = bounded - period_cost
        return (1.0 + hedged_sub) ** substeps - 1.0

    if hedging_config.strategy == "covered_call":
        write_fraction = float(np.clip(hedging_config.covered_call_write_fraction, 0.0, 1.0))
        strike_sub = (1.0 + hedging_config.covered_call_strike_otm) ** (1.0 / substeps) - 1.0
        premium_sub = hedging_config.covered_call_premium_annual / cost_periods_per_year
        assignment_cost_sub = hedging_config.covered_call_assignment_cost / substeps
        written_leg = premium_sub + min(raw_subperiod_return, strike_sub)
        if raw_subperiod_return > strike_sub:
            written_leg -= assignment_cost_sub
        hedged_sub = (1.0 - write_fraction) * raw_subperiod_return + write_fraction * written_leg
        return (1.0 + hedged_sub) ** substeps - 1.0

    raise ValueError(f"Unsupported hedging strategy: {hedging_config.strategy}")


def _apply_hedge_to_return_array(
    returns: np.ndarray,
    hedging_config: HedgingConfig,
    periods_per_year: int,
) -> np.ndarray:
    """
    Vectorized equivalent of apply_hedge_to_equity_return for performance.
    """
    if not hedging_config.enabled or hedging_config.strategy == "none":
        return returns
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    if hedging_config.rehedge_substeps_per_period <= 0:
        raise ValueError(
            "rehedge_substeps_per_period must be positive, "
            f"got {hedging_config.rehedge_substeps_per_period}"
        )

    substeps = hedging_config.rehedge_substeps_per_period
    cost_periods_per_year = periods_per_year * substeps
    raw_subperiod_returns = np.power(np.maximum(1.0 + returns, 0.0), 1.0 / substeps) - 1.0

    if hedging_config.strategy == "protective_put":
        period_cost = hedging_config.protective_put_cost_annual / cost_periods_per_year
        floor_sub = (1.0 + hedging_config.protective_put_floor) ** (1.0 / substeps) - 1.0
        coverage = float(np.clip(hedging_config.protective_put_coverage, 0.0, 1.0))
        hedged_sub = raw_subperiod_returns - period_cost
        mask = raw_subperiod_returns < floor_sub
        if np.any(mask):
            hedged_sub = hedged_sub.copy()
            hedged_sub[mask] += coverage * (floor_sub - raw_subperiod_returns[mask])
        return np.power(np.maximum(1.0 + hedged_sub, 0.0), substeps) - 1.0

    if hedging_config.strategy == "tail_hedge":
        period_cost = hedging_config.tail_hedge_cost_annual / cost_periods_per_year
        trigger_sub = (1.0 + hedging_config.tail_hedge_trigger) ** (1.0 / substeps) - 1.0
        base = raw_subperiod_returns - period_cost
        mask = raw_subperiod_returns < trigger_sub
        if not np.any(mask):
            return np.power(np.maximum(1.0 + base, 0.0), substeps) - 1.0
        excess_drawdown = trigger_sub - raw_subperiod_returns[mask]
        base = base.copy()
        base[mask] += hedging_config.tail_hedge_slope * excess_drawdown
        return np.power(np.maximum(1.0 + base, 0.0), substeps) - 1.0

    if hedging_config.strategy == "collar":
        period_cost = hedging_config.collar_cost_annual / cost_periods_per_year
        floor_sub = (1.0 + hedging_config.collar_floor) ** (1.0 / substeps) - 1.0
        cap_sub = (1.0 + hedging_config.collar_cap) ** (1.0 / substeps) - 1.0
        bounded = np.minimum(
            cap_sub,
            np.maximum(raw_subperiod_returns, floor_sub),
        )
        hedged_sub = bounded - period_cost
        return np.power(np.maximum(1.0 + hedged_sub, 0.0), substeps) - 1.0

    if hedging_config.strategy == "covered_call":
        write_fraction = float(np.clip(hedging_config.covered_call_write_fraction, 0.0, 1.0))
        strike_sub = (1.0 + hedging_config.covered_call_strike_otm) ** (1.0 / substeps) - 1.0
        premium_sub = hedging_config.covered_call_premium_annual / cost_periods_per_year
        assignment_cost_sub = hedging_config.covered_call_assignment_cost / substeps
        written_leg = premium_sub + np.minimum(raw_subperiod_returns, strike_sub)
        assigned_mask = raw_subperiod_returns > strike_sub
        if np.any(assigned_mask):
            written_leg = written_leg.copy()
            written_leg[assigned_mask] -= assignment_cost_sub
        hedged_sub = (1.0 - write_fraction) * raw_subperiod_returns + write_fraction * written_leg
        return np.power(np.maximum(1.0 + hedged_sub, 0.0), substeps) - 1.0

    raise ValueError(f"Unsupported hedging strategy: {hedging_config.strategy}")


def load_shiller_annual_stock_bond_returns(
    file_path: str = "data/ie_data.xls",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load aligned annual nominal stock and bond returns from Shiller ie_data.xls.

    Stock series:
      Reconstructed nominal total-return index from Real TR Price * CPI, then
      annual pct_change of year-end values.
    Bond series:
      Monthly long-term bond gross return relatives (Shiller first "Returns"
      column) compounded within each year, minus 1.

    Returns:
        (stock_returns, bond_returns) aligned year-for-year.
    """
    raw = pd.read_excel(file_path, sheet_name="Data", skiprows=7, header=None)
    headers = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = headers

    df["Date"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df["CPI"] = pd.to_numeric(df.iloc[:, 4], errors="coerce")
    df["RealTotalReturnPrice"] = pd.to_numeric(df.iloc[:, 9], errors="coerce")
    # First "Returns" column is monthly gross long-bond total return relative.
    df["BondGrossMonthly"] = pd.to_numeric(df.iloc[:, 17], errors="coerce")
    df["Year"] = np.floor(df["Date"]).astype("Int64")

    stock_df = df.dropna(subset=["Year", "CPI", "RealTotalReturnPrice"]).copy()
    stock_df["NominalTotalReturnPrice"] = (
        stock_df["RealTotalReturnPrice"] * stock_df["CPI"]
    )
    stock_annual = (
        stock_df.groupby("Year")["NominalTotalReturnPrice"].last().dropna().pct_change()
    )

    bond_df = df.dropna(subset=["Year", "BondGrossMonthly"]).copy()
    bond_annual = bond_df.groupby("Year")["BondGrossMonthly"].prod().dropna() - 1.0

    aligned_years = stock_annual.dropna().index.intersection(bond_annual.index)
    if len(aligned_years) == 0:
        raise ValueError("No overlapping annual stock/bond history found in ie_data.xls")

    stock_returns = stock_annual.loc[aligned_years].to_numpy(dtype=np.float64)
    bond_returns = bond_annual.loc[aligned_years].to_numpy(dtype=np.float64)
    return stock_returns, bond_returns


def load_shiller_annual_returns(
    file_path: str = "data/ie_data.xls",
    *,
    nominal: bool = True,
) -> np.ndarray:
    """
    Load annual S&P Composite total-return series from Shiller's ie_data.xls.

    The engine is run in NOMINAL dollars (nominal=True, the default), so that
    a $100K withdrawal today grows with inflation to ~$400K in 40 years, and
    stock/bond returns are also nominal. This keeps every simulation in the
    same units.

    Shiller's file has a "Real Total Return Price" column (CPI-adjusted,
    dividends reinvested) but no explicit nominal counterpart. We rebuild
    nominal total return by multiplying by CPI; the base-period CPI constant
    drops out when we take pct_change.

    Args:
        file_path: path to Shiller's ie_data.xls
        nominal: if True (default), returns nominal annual returns. If False,
            returns real (CPI-adjusted) annual returns — kept for reference /
            backwards compatibility with prior behavior.

    Returns:
        1D numpy array of annual returns, one per year.
    """
    df = pd.read_excel(file_path, sheet_name="Data", skiprows=8)
    # Col 0: Date (e.g. 1871.01), Col 4: CPI, Col 9: Real Total Return Price
    df = df.iloc[:, [0, 4, 9]]
    df.columns = ["Date", "CPI", "RealTotalReturnPrice"]
    # Shiller's file has a few trailing rows with 'NA' strings / object dtype.
    # Coerce to numeric and drop those rows so pandas doesn't emit
    # FutureWarnings about implicit object-dtype downcasting in pct_change.
    df["CPI"] = pd.to_numeric(df["CPI"], errors="coerce")
    df["RealTotalReturnPrice"] = pd.to_numeric(
        df["RealTotalReturnPrice"], errors="coerce"
    )
    df = df.dropna(subset=["Date", "CPI", "RealTotalReturnPrice"])
    df["Year"] = df["Date"].astype(str).str.split(".").str[0].astype(int)

    if nominal:
        # Nominal_TR[t] = Real_TR[t] * CPI[t] (up to a constant base-CPI
        # factor that cancels in pct_change).
        df["Series"] = df["RealTotalReturnPrice"] * df["CPI"]
    else:
        df["Series"] = df["RealTotalReturnPrice"]

    annual = df.groupby("Year")["Series"].last().dropna()
    return annual.pct_change().dropna().to_numpy()


def format_withdrawal_breakdown(
    *,
    withdrawal: float,
    withdrawal_negative_year: float,
    wife_supplemental_income: float,
    wife_years_with_supplemental_income: int,
    me_supplemental_income: float,
    me_years_with_supplemental_income: int,
    social_security_money: float,
    years_without_social_security: int,
    n_years: int,
) -> list[str]:
    """Return lines describing net portfolio withdrawals by retirement period."""
    boundaries: set[int] = {1, n_years + 1}
    if wife_supplemental_income > 0 and wife_years_with_supplemental_income < n_years:
        boundaries.add(wife_years_with_supplemental_income + 1)
    if me_supplemental_income > 0 and me_years_with_supplemental_income < n_years:
        boundaries.add(me_years_with_supplemental_income + 1)
    if social_security_money > 0 and years_without_social_security < n_years:
        boundaries.add(years_without_social_security + 1)

    sorted_bounds = sorted(boundaries)
    show_neg = withdrawal_negative_year != withdrawal

    lines: list[str] = []
    lines.append("  Portfolio Withdrawal Breakdown (before inflation):")

    for i in range(len(sorted_bounds) - 1):
        yr_start = sorted_bounds[i]
        yr_end = min(sorted_bounds[i + 1] - 1, n_years)
        if yr_start > n_years:
            break

        has_wife_supp = (
            wife_supplemental_income > 0
            and yr_start <= wife_years_with_supplemental_income
        )
        has_me_supp = (
            me_supplemental_income > 0
            and yr_start <= me_years_with_supplemental_income
        )
        has_ss = social_security_money > 0 and yr_start > years_without_social_security

        income_parts: list[str] = []
        total_income = 0.0
        if has_wife_supp:
            income_parts.append(f"${wife_supplemental_income:,.0f} wife supplemental")
            total_income += wife_supplemental_income
        if has_me_supp:
            income_parts.append(f"${me_supplemental_income:,.0f} me supplemental")
            total_income += me_supplemental_income
        if has_ss:
            income_parts.append(f"${social_security_money:,.0f} SS")
            total_income += social_security_money

        net = max(withdrawal - total_income, 0)
        net_neg = max(withdrawal_negative_year - total_income, 0)

        period = (
            f"Years {yr_start}-{yr_end}"
            if yr_start != yr_end
            else f"Year {yr_start}"
        )

        if total_income > 0:
            income_str = " + ".join(income_parts)
            line = f"    {period}: ${withdrawal:,.0f} - {income_str} = ${net:,.0f} from portfolio"
            if show_neg:
                line += f" (neg yr: ${net_neg:,.0f})"
        else:
            line = f"    {period}: ${withdrawal:,.0f} from portfolio"
            if show_neg:
                line += f" (neg yr: ${withdrawal_negative_year:,.0f})"

        lines.append(line)

    return lines


# global to be set in initializer
_RETURNS: np.ndarray
_N_YEARS: int 
_INITIAL_BALANCE: float
_WITHDRAWAL: float 
_WITHDRAWAL_NEGATIVE_YEAR: float 
_GO_BACK_YEARS: int
_SAMPLING_MODE: str
_BLOCK_BOOTSTRAP_SIZE: int
_SP500_PERCENTAGE: float 
_BOND_RATE: float 
_BOND_RETURNS: Optional[np.ndarray]
_BOND_RETURN_MODE: BondReturnMode
_INFLATION_RATE: float 
_YEARS_WITHOUT_SOCIAL_SECURITY: int 
_SOCIAL_SECURITY_MONEY: float
_WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME: int
_WIFE_SUPPLEMENTAL_INCOME: float
_ME_YEARS_WITH_SUPPLEMENTAL_INCOME: int
_ME_SUPPLEMENTAL_INCOME: float
_HEDGING_CONFIG: HedgingConfig
_HEDGE_PERIODS_PER_YEAR: int
_ANNUAL_EXPENSE_RATIO: float


def _init_worker(
    returns: np.ndarray,
    bond_returns: Optional[np.ndarray],
    n_years: int,
    initial_balance: float,
    withdrawal: float,
    withdrawal_negative_year: float,
    go_back_year: int,
    sampling_mode: str,
    block_bootstrap_size: int,
    sp500_percentage: float,
    bond_rate: float,
    bond_return_mode: BondReturnMode,
    inflation_rate: float,
    years_without_social_security: int,
    social_security_money: float,
    wife_years_with_supplemental_income: int,
    wife_supplemental_income: float,
    me_years_with_supplemental_income: int,
    me_supplemental_income: float,
    hedging_config: HedgingConfig,
    annual_expense_ratio: float,
):
    global _RETURNS, _BOND_RETURNS, _N_YEARS, _INITIAL_BALANCE, _WITHDRAWAL, _WITHDRAWAL_NEGATIVE_YEAR, _GO_BACK_YEARS, \
        _SAMPLING_MODE, _BLOCK_BOOTSTRAP_SIZE, _SP500_PERCENTAGE, _BOND_RATE, _BOND_RETURN_MODE, _INFLATION_RATE, \
        _YEARS_WITHOUT_SOCIAL_SECURITY, _SOCIAL_SECURITY_MONEY, \
        _WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME, _WIFE_SUPPLEMENTAL_INCOME, \
        _ME_YEARS_WITH_SUPPLEMENTAL_INCOME, _ME_SUPPLEMENTAL_INCOME, \
        _HEDGING_CONFIG, _HEDGE_PERIODS_PER_YEAR, _ANNUAL_EXPENSE_RATIO
    _RETURNS = returns
    _BOND_RETURNS = bond_returns
    _N_YEARS = n_years
    _INITIAL_BALANCE = initial_balance
    _WITHDRAWAL = withdrawal
    _WITHDRAWAL_NEGATIVE_YEAR = withdrawal_negative_year
    _GO_BACK_YEARS = go_back_year
    _SAMPLING_MODE = sampling_mode
    _BLOCK_BOOTSTRAP_SIZE = block_bootstrap_size
    _SP500_PERCENTAGE = sp500_percentage
    _BOND_RATE = bond_rate
    _BOND_RETURN_MODE = bond_return_mode
    _INFLATION_RATE = inflation_rate
    _YEARS_WITHOUT_SOCIAL_SECURITY = years_without_social_security
    _SOCIAL_SECURITY_MONEY = social_security_money
    _WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME = wife_years_with_supplemental_income
    _WIFE_SUPPLEMENTAL_INCOME = wife_supplemental_income
    _ME_YEARS_WITH_SUPPLEMENTAL_INCOME = me_years_with_supplemental_income
    _ME_SUPPLEMENTAL_INCOME = me_supplemental_income
    _HEDGING_CONFIG = hedging_config
    _HEDGE_PERIODS_PER_YEAR = periods_per_year_from_frequency(
        hedging_config.rebalance_frequency
    )
    _ANNUAL_EXPENSE_RATIO = float(annual_expense_ratio)

def _simulate_chunk(args):
    """
    Worker: simulate `chunk_size` independent simulations.
    Returns final_balances and (optionally) trajectories.

    args: (chunk_size, seed, return_trajectories_bool)
    """
    chunk_size, seed, return_traj = args
    rng = np.random.default_rng(seed)
    m = len(_RETURNS)

    # Generate per-scenario index sequences according to sampling mode.
    #   - "block_bootstrap": preserves multi-year crash regimes (recommended)
    #   - "constrained":     bounds consecutive losing/winning streaks
    #   - "random":          single-year permutation w/o replacement (legacy)
    idx = np.empty((chunk_size, _N_YEARS), dtype=np.int64)
    if _SAMPLING_MODE == "block_bootstrap":
        for i in range(chunk_size):
            idx[i] = generate_block_bootstrap_indices(
                rng, m, _N_YEARS, block_size=_BLOCK_BOOTSTRAP_SIZE
            )
    elif _SAMPLING_MODE == "constrained":
        for i in range(chunk_size):
            idx[i] = generate_constrained_indices(rng, _RETURNS, _N_YEARS)
    elif _SAMPLING_MODE == "random":
        for i in range(chunk_size):
            idx[i] = rng.permutation(m)[:_N_YEARS]
    else:
        raise ValueError(f"Unknown sampling_mode: {_SAMPLING_MODE!r}")

    # Get the returns for each simulation
    sim_returns = _RETURNS[idx]  # shape (chunk_size, _N_YEARS)
    if _BOND_RETURN_MODE == "historical":
        if _BOND_RETURNS is None:
            raise ValueError("bond_return_mode='historical' requires bond return series")
        sim_bond_returns = _BOND_RETURNS[idx]
    else:
        sim_bond_returns = None

    # Initialize balances and optional trajectories
    balances = np.full(chunk_size, _INITIAL_BALANCE, dtype=np.float64)
    if return_traj:
        trajectories = np.zeros((chunk_size, _N_YEARS + 1), dtype=np.float64)
        trajectories[:, 0] = _INITIAL_BALANCE
    else:
        trajectories = None

    # Precompute inflation factors for each year
    inflation_factors = (1.0 + _INFLATION_RATE) ** np.arange(_N_YEARS)

    # Ensure _SP500_PERCENTAGE is between 0 and 1
    sp500_frac = np.clip(_SP500_PERCENTAGE, 0.0, 1.0)
    bond_frac = 1.0 - sp500_frac

    # Determine the amount of social security to add each year
    social_security_per_year = np.zeros(_N_YEARS, dtype=np.float64)
    if _SOCIAL_SECURITY_MONEY > 0:
        years_with_ss = max(0, _N_YEARS - _YEARS_WITHOUT_SOCIAL_SECURITY)
        # Only assign when there are actual years with social security.
        if years_with_ss > 0:
            social_security_per_year[-years_with_ss:] = _SOCIAL_SECURITY_MONEY * inflation_factors[-years_with_ss:]  # Adjust for inflation

    # Add supplemental income if applicable which is reduced from the withdrawal.
    # Wife and me are tracked independently so each can have a different
    # duration and amount.
    supplemental_income_per_year = np.zeros(_N_YEARS, dtype=np.float64)
    if _WIFE_SUPPLEMENTAL_INCOME > 0:
        wife_years = min(_WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME, _N_YEARS)
        supplemental_income_per_year[:wife_years] += (
            _WIFE_SUPPLEMENTAL_INCOME * inflation_factors[:wife_years]
        )
    if _ME_SUPPLEMENTAL_INCOME > 0:
        me_years = min(_ME_YEARS_WITH_SUPPLEMENTAL_INCOME, _N_YEARS)
        supplemental_income_per_year[:me_years] += (
            _ME_SUPPLEMENTAL_INCOME * inflation_factors[:me_years]
        )

    # Vectorized simulation over years
    prev_portfolio_return: Optional[np.ndarray] = None
    for t in range(_N_YEARS):
        # Year 1 has no realized prior-year return, so default to regular withdrawal.
        if t == 0 or prev_portfolio_return is None:
            withdrawals = np.full(chunk_size, float(_WITHDRAWAL), dtype=np.float64)
        else:
            # Use prior-year realized portfolio return to avoid look-ahead bias.
            withdrawals = np.where(
                prev_portfolio_return >= 0.0,
                float(_WITHDRAWAL),
                float(_WITHDRAWAL_NEGATIVE_YEAR),
            )
        withdrawals *= inflation_factors[t]  # Apply inflation
        # Reduce portfolio withdrawal by social security for this year (do not double-count).
        ss_amount = social_security_per_year[t]
        # Reduce portfolio withdrawal by supplemental income for this year
        supp_amount = supplemental_income_per_year[t]
        # Net withdrawal from portfolio (cannot be negative).
        withdrawals = np.maximum(withdrawals - ss_amount - supp_amount, 0.0)

        equity_returns_t = sim_returns[:, t]
        if _BOND_RETURN_MODE == "historical":
            assert sim_bond_returns is not None
            bond_returns_t = sim_bond_returns[:, t]
        else:
            bond_returns_t = _BOND_RATE

        if _HEDGING_CONFIG.apply_to_equity_only:
            hedged_equity_returns = _apply_hedge_to_return_array(
                equity_returns_t,
                _HEDGING_CONFIG,
                _HEDGE_PERIODS_PER_YEAR,
            )
            portfolio_return = sp500_frac * hedged_equity_returns + bond_frac * bond_returns_t
        else:
            blended_return = sp500_frac * equity_returns_t + bond_frac * bond_returns_t
            portfolio_return = _apply_hedge_to_return_array(
                blended_return,
                _HEDGING_CONFIG,
                _HEDGE_PERIODS_PER_YEAR,
            )

        portfolio_return = portfolio_return - _ANNUAL_EXPENSE_RATIO

        # Compute portfolio growth using linear allocation
        portfolio_growth = 1 + portfolio_return

        # Update balances after withdrawals and growth
        balances = (balances - withdrawals) * portfolio_growth

        # Reset balances to initial if they drop below the initial balance in early years
        if t < _GO_BACK_YEARS:
            under_initial = balances < _INITIAL_BALANCE
            balances[under_initial] = _INITIAL_BALANCE

        # Apply floor at zero
        np.maximum(balances, 0.0, out=balances)

        if return_traj:
            trajectories[:, t + 1] = balances
        prev_portfolio_return = portfolio_return

    return balances, trajectories



def run_simulation_mp(
    n_sims: int = 100_000,
    n_years: int = 35,
    initial_balance: float = 6_000_000,
    withdrawal: float = 100_000,
    withdrawal_negative_year: float = 100_000,
    go_back_year: int = 0,
    n_workers: Optional[int] = None,
    return_trajectories: bool=False,
    chunk_size: Optional[int] = None,
    random_with_real_life_constraints: bool = False,
    sampling_mode: Optional[str] = None,
    block_bootstrap_size: int = 5,
    sp500_percentage: float = 1.0,
    bond_rate: float = 0.04,
    bond_return_mode: BondReturnMode = "fixed",
    inflation_rate: float = 0.03,
    years_without_social_security: int = 35,
    social_security_money: float = 0,
    wife_years_with_supplemental_income: int = 0,
    wife_supplemental_income: float = 0,
    me_years_with_supplemental_income: int = 0,
    me_supplemental_income: float = 0,
    random_seed: Optional[int] = None,
    hedging_config: Optional[HedgingConfig] = None,
    returns_override: Optional[np.ndarray] = None,
    bond_returns_override: Optional[np.ndarray] = None,
    annual_expense_ratio: float = 0.0,
    reserve_floor: Optional[float] = None,
):
    """
    Run Monte Carlo retirement portfolio simulations using multiprocessing.

    UNITS: Everything is in NOMINAL (today-forward) dollars.
    - `withdrawal`, `social_security_money`, `wife_supplemental_income`,
      `me_supplemental_income`, and `initial_balance` are entered as TODAY's
      dollars.
    - Withdrawals, SS, and supplemental income all grow with `inflation_rate`.
    - Stock returns are NOMINAL (reconstructed from Shiller Real TR × CPI).
    - `bond_rate` is NOMINAL.
    - So a $100K withdrawal today grows to roughly $100K * (1+inflation)^40
      by year 40, while the portfolio also grows in nominal terms.

    KEY ASSUMPTIONS:
    1. Withdrawals occur at the START of each year (before portfolio growth)
       This is more conservative than withdrawing at year-end.

    2. Portfolio allocation is LINEAR and rebalanced annually:
       return = sp500_percentage * stock_return + (1 - sp500_percentage) * bond_return
       This assumes perfect rebalancing to target allocation each year.

    3. Bond returns can be:
       - fixed (bond_return_mode="fixed"): bond_rate each year, or
       - historical (bond_return_mode="historical"): sampled from Shiller's
         long-bond return series using the same sampled years as equities.

    4. All amounts (withdrawals, income) are adjusted for inflation annually
       using compound inflation: amount * (1 + inflation_rate)^year

    5. Income sources reduce portfolio withdrawals (not added to balance)
       Net withdrawal = max(
           0,
           withdrawal
           - social_security
           - wife_supplemental_income
           - me_supplemental_income,
       )
       Wife and me supplemental income are tracked independently so each can
       have a different duration and amount.

    5b. Negative-year withdrawal switching uses PRIOR-YEAR realized portfolio
        return sign (year 1 defaults to regular withdrawal), avoiding look-ahead.

    6. Sampling modes (sampling_mode parameter):
       - "block_bootstrap" (DEFAULT, recommended): Samples consecutive blocks of
         `block_bootstrap_size` historical years (with replacement, circular).
         Preserves multi-year crash sequences (1929-32, 1973-74, 2000-02,
         2007-09) that drive sequence-of-returns risk in real retirements.
       - "random": Single-year permutation without replacement. Destroys
         autocorrelation; tends to under-state sequence risk.
       - "constrained": Bounds consecutive negative/positive years and
         cumulative changes to avoid pathological streaks.

       Backwards-compat: `random_with_real_life_constraints=True` is mapped
       to "constrained" if `sampling_mode` is not explicitly set.

    Parameters:
        n_sims: Number of simulation runs
        n_years: Retirement duration in years
        initial_balance: Starting portfolio value ($)
        withdrawal: Annual withdrawal in positive market years ($)
        withdrawal_negative_year: Annual withdrawal in negative market years ($)
        go_back_year: Number of early years where balance resets to initial if it drops below
        n_workers: Number of worker processes (default: cpu_count - 1)
        return_trajectories: If True, return full year-by-year trajectories
        chunk_size: Simulations per worker chunk (default: auto-calculated)
        random_with_real_life_constraints: Legacy flag. If sampling_mode is
            None, True maps to "constrained" and False to "block_bootstrap".
        sampling_mode: One of {"block_bootstrap", "random", "constrained"}.
            If None (default), derived from random_with_real_life_constraints.
        block_bootstrap_size: Block length (years) when sampling_mode is
            "block_bootstrap". Default 5.
        sp500_percentage: Fraction of portfolio in stocks (0.0 to 1.0)
        bond_rate: Annual bond return rate (e.g., 0.04 for 4%) when
            bond_return_mode="fixed".
        bond_return_mode: "fixed" (default) or "historical".
        inflation_rate: Annual inflation rate (e.g., 0.03 for 3%)
        years_without_social_security: Years until social security starts
        social_security_money: Annual social security income ($, before inflation)
        wife_years_with_supplemental_income: Years the wife earns supplemental income
        wife_supplemental_income: Annual wife supplemental income ($, before inflation)
        me_years_with_supplemental_income: Years I earn supplemental income
        me_supplemental_income: Annual me supplemental income ($, before inflation)
        random_seed: Optional seed for reproducibility (None = random)
        hedging_config: Optional hedge return-shaping config. This is an approximation,
            not executable options pricing or options contract simulation.
        returns_override: Optional equity return series override.
        bond_returns_override: Optional bond return series override for
            bond_return_mode="historical". Must match equity series length.
        annual_expense_ratio: Annual advisory + fund expense drag subtracted from
            each year's portfolio return (simple haircut; not tax-aware).
        reserve_floor: If set, report P(final balance < this floor) on SimulationData.

    Returns:
        SimulationData object containing results, statistics, and optional trajectories
    """
    bond_returns: Optional[np.ndarray] = None
    if bond_return_mode not in ("fixed", "historical"):
        raise ValueError(
            f"bond_return_mode must be one of 'fixed', 'historical'; got {bond_return_mode!r}"
        )

    if returns_override is not None:
        returns = np.asarray(returns_override, dtype=np.float64)
        if returns.ndim != 1:
            raise ValueError(
                "returns_override must be a 1D array of period returns, "
                f"got shape {returns.shape}"
            )
        if len(returns) == 0:
            raise ValueError("returns_override cannot be empty")
        if bond_return_mode == "historical":
            if bond_returns_override is None:
                raise ValueError(
                    "bond_return_mode='historical' with returns_override requires "
                    "bond_returns_override of equal length"
                )
            bond_returns = np.asarray(bond_returns_override, dtype=np.float64)
            if bond_returns.ndim != 1:
                raise ValueError(
                    "bond_returns_override must be a 1D array of period returns, "
                    f"got shape {bond_returns.shape}"
                )
            if len(bond_returns) != len(returns):
                raise ValueError(
                    "bond_returns_override length must match returns_override length, "
                    f"got {len(bond_returns)} vs {len(returns)}"
                )
    else:
        # NOMINAL returns so that stock returns, bond rate, and inflated
        # withdrawals are all in the same (nominal) units.
        if bond_return_mode == "historical":
            returns, bond_returns = load_shiller_annual_stock_bond_returns()
        else:
            returns = load_shiller_annual_returns(nominal=True)
    total_years = len(returns)

    # Input validation
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")
    if n_years <= 0:
        raise ValueError(f"n_years must be positive, got {n_years}")
    if n_years > total_years:
        raise ValueError(
            f"n_years ({n_years}) cannot exceed available historical data ({total_years} years)"
        )
    if initial_balance <= 0:
        raise ValueError(f"initial_balance must be positive, got {initial_balance}")
    if withdrawal < 0:
        raise ValueError(f"withdrawal cannot be negative, got {withdrawal}")
    if withdrawal_negative_year < 0:
        raise ValueError(
            f"withdrawal_negative_year cannot be negative, got {withdrawal_negative_year}"
        )
    if withdrawal_negative_year > withdrawal:
        warnings.warn(
            f"withdrawal_negative_year ({withdrawal_negative_year}) > withdrawal ({withdrawal}). "
            "This means withdrawing MORE in down markets, which is unusual."
        )
    if not 0 <= sp500_percentage <= 1:
        raise ValueError(
            f"sp500_percentage must be between 0 and 1, got {sp500_percentage}"
        )
    if go_back_year < 0:
        raise ValueError(f"go_back_year cannot be negative, got {go_back_year}")
    if years_without_social_security < 0:
        raise ValueError(
            f"years_without_social_security cannot be negative, got {years_without_social_security}"
        )
    if social_security_money < 0:
        raise ValueError(
            f"social_security_money cannot be negative, got {social_security_money}"
        )
    if wife_years_with_supplemental_income < 0:
        raise ValueError(
            "wife_years_with_supplemental_income cannot be negative, "
            f"got {wife_years_with_supplemental_income}"
        )
    if wife_supplemental_income < 0:
        raise ValueError(
            f"wife_supplemental_income cannot be negative, got {wife_supplemental_income}"
        )
    if me_years_with_supplemental_income < 0:
        raise ValueError(
            "me_years_with_supplemental_income cannot be negative, "
            f"got {me_years_with_supplemental_income}"
        )
    if me_supplemental_income < 0:
        raise ValueError(
            f"me_supplemental_income cannot be negative, got {me_supplemental_income}"
        )
    if annual_expense_ratio < 0.0:
        raise ValueError(
            f"annual_expense_ratio cannot be negative, got {annual_expense_ratio}"
        )
    if annual_expense_ratio >= 1.0:
        raise ValueError(
            f"annual_expense_ratio must be < 1.0 (fraction of assets), got {annual_expense_ratio}"
        )
    if reserve_floor is not None and reserve_floor < 0:
        raise ValueError(f"reserve_floor must be >= 0 when set, got {reserve_floor}")

    if go_back_year > 0:
        warnings.warn(
            f"go_back_year={go_back_year}: balances reset to initial_balance when "
            f"below initial in the first {go_back_year} years — this is not a standard "
            f"retirement assumption and can inflate success rates.",
            UserWarning,
            stacklevel=2,
        )

    if hedging_config is None:
        hedging_config = HedgingConfig(
            enabled=False,
            strategy="none",
            rebalance_frequency="yearly",
        )

    # Resolve sampling mode. Block bootstrap is the new default because it
    # preserves multi-year crash regimes (sequence-of-returns risk).
    if sampling_mode is None:
        sampling_mode = (
            "constrained" if random_with_real_life_constraints else "block_bootstrap"
        )
    if sampling_mode not in ("random", "constrained", "block_bootstrap"):
        raise ValueError(
            f"sampling_mode must be one of 'random', 'constrained', "
            f"'block_bootstrap'; got {sampling_mode!r}"
        )
    if block_bootstrap_size <= 0:
        raise ValueError(
            f"block_bootstrap_size must be positive, got {block_bootstrap_size}"
        )

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # leave one core for OS/other tasks

    if chunk_size is None:
        # choose chunk_size so each worker gets a few thousand sims
        # avoid too many small tasks; tune for your memory
        chunk_size = max(1_000, n_sims // (n_workers * 4))

    # Create the task list of (chunk_size, seed, return_trajectories)
    tasks = []
    remaining = n_sims

    # Use provided seed for reproducibility, or generate random seed
    if random_seed is not None:
        seed_base = random_seed
    else:
        seed_base = int(np.random.default_rng().integers(0, 2**31 - 1))

    worker_seed = int(seed_base) & 0x7FFFFFFF
    i = 0
    while remaining > 0:
        c = min(chunk_size, remaining)
        tasks.append((c, worker_seed + i, return_trajectories))
        remaining -= c
        i += 1

    # Start pool with an initializer that sets global returns in each worker
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            returns,
            bond_returns,
            n_years,
            initial_balance,
            withdrawal,
            withdrawal_negative_year,
            go_back_year,
            sampling_mode,
            block_bootstrap_size,
            sp500_percentage,
            bond_rate,
            bond_return_mode,
            inflation_rate,
            years_without_social_security,
            social_security_money,
            wife_years_with_supplemental_income,
            wife_supplemental_income,
            me_years_with_supplemental_income,
            me_supplemental_income,
            hedging_config,
            annual_expense_ratio,
        ),
    ) as pool:
        results = pool.map(_simulate_chunk, tasks)

    # Combine results
    final_balances_list = []
    if return_trajectories:
        traj_list = []
    for balances, traj in results:
        final_balances_list.append(balances)
        if return_trajectories:
            traj_list.append(traj)

    final_balances = np.concatenate(final_balances_list)
    if return_trajectories:
        trajectories = np.concatenate(traj_list, axis=0)
    else:
        trajectories = None

    # If you produced more sims due to chunking rounding, trim to n_sims
    final_balances = final_balances[:n_sims]
    if return_trajectories and trajectories is not None:
        trajectories = trajectories[:n_sims]

    # end_time = time.time()
    # mode = (
    #     "constrained random"
    #     if random_with_real_life_constraints
    #     else "unconstrained random"
    # )
    # print(
    #     f"Simulation of {n_sims:,} runs over {n_years} years took {end_time - start_time:.2f} seconds using {mode}."
    # )

    # Return a simple object (adapt to your SimulationData)
    return SimulationData(
        initial_balance,
        withdrawal,
        withdrawal_negative_year,
        n_years,
        n_sims,
        final_balances,
        trajectories,
        total_years,
        returns,
        go_back_year=go_back_year,
        reserve_floor=reserve_floor,
        sampling_mode=str(sampling_mode),
        block_bootstrap_size=block_bootstrap_size,
        bond_return_mode=str(bond_return_mode),
        inflation_rate=inflation_rate,
        bond_rate=bond_rate,
        annual_expense_ratio=annual_expense_ratio,
        overlapping_windows_note=None,
    )


def run_simulation_historical_real(
    n_years=45,
    initial_balance=4_800_000,
    withdrawal=120_000,
    withdrawal_negative_year=80_000,
    go_back_year=0,
    return_trajectories=False,
    inflation_rate=0.03,
    sp500_percentage=1.0,
    bond_rate=0.04,
    bond_return_mode: BondReturnMode = "fixed",
    years_without_social_security=45,
    social_security_money=0,
    wife_years_with_supplemental_income=0,
    wife_supplemental_income=0,
    me_years_with_supplemental_income=0,
    me_supplemental_income=0,
    hedging_config: Optional[HedgingConfig] = None,
    annual_expense_ratio: float = 0.0,
    reserve_floor: Optional[float] = None,
):
    """
    Run historical rolling-window backtesting using actual market sequences.

    Uses every possible n_year window from historical data to test portfolio survival.
    For example, with 150 years of data and n_years=45, runs 106 simulations
    (starting years 1871, 1872, ..., 1976 if data ends in 2020).

    This tests "what would have happened if you retired in year X" for all X.
    Uses the SAME assumptions as run_simulation_mp() for consistency.

    Identifies which historical starting years would have led to portfolio failure,
    providing concrete examples of challenging market environments.

    Parameters:
        (Same as run_simulation_mp, except no random_seed since order is deterministic)

    Returns:
        SimulationData object with results from all rolling windows
    """
    start_time = time.time()

    if annual_expense_ratio < 0.0:
        raise ValueError(
            f"annual_expense_ratio cannot be negative, got {annual_expense_ratio}"
        )
    if annual_expense_ratio >= 1.0:
        raise ValueError(
            f"annual_expense_ratio must be < 1.0, got {annual_expense_ratio}"
        )
    if reserve_floor is not None and reserve_floor < 0:
        raise ValueError(f"reserve_floor must be >= 0 when set, got {reserve_floor}")
    if go_back_year > 0:
        warnings.warn(
            f"go_back_year={go_back_year}: balances reset to initial_balance when "
            f"below initial in the first {go_back_year} years — not a standard "
            f"retirement assumption.",
            UserWarning,
            stacklevel=2,
        )

    if hedging_config is None:
        hedging_config = HedgingConfig(
            enabled=False,
            strategy="none",
            rebalance_frequency="yearly",
        )
    periods_per_year = periods_per_year_from_frequency(
        hedging_config.rebalance_frequency
    )

    # --- Load and prepare historical data (NOMINAL, same as run_simulation_mp) ---
    file_path = "data/ie_data.xls"
    if bond_return_mode == "historical":
        returns, bond_returns = load_shiller_annual_stock_bond_returns(file_path=file_path)
    elif bond_return_mode == "fixed":
        returns = load_shiller_annual_returns(file_path=file_path, nominal=True)
        bond_returns = None
    else:
        raise ValueError(
            f"bond_return_mode must be one of 'fixed', 'historical'; got {bond_return_mode!r}"
        )
    total_years = len(returns)

    # --- Setup rolling window simulations ---
    n_sims = total_years - n_years + 1
    start_indices = np.arange(n_sims)

    final_balances = np.zeros(n_sims, dtype=np.float64)
    trajectories = (
        np.zeros((n_sims, n_years + 1), dtype=np.float64)
        if return_trajectories
        else None
    )

    # Precompute inflation factors for each year
    inflation_factors = (1.0 + inflation_rate) ** np.arange(n_years)

    # Ensure sp500_percentage is between 0 and 1
    sp500_frac = np.clip(sp500_percentage, 0.0, 1.0)
    bond_frac = 1.0 - sp500_frac

    # Determine the amount of social security to add each year
    social_security_per_year = np.zeros(n_years, dtype=np.float64)
    if social_security_money > 0:
        years_with_ss = max(0, n_years - years_without_social_security)
        if years_with_ss > 0:
            social_security_per_year[-years_with_ss:] = social_security_money * inflation_factors[-years_with_ss:]

    # Add supplemental income if applicable. Wife and me are tracked
    # independently so each can have a different duration and amount.
    supplemental_income_per_year = np.zeros(n_years, dtype=np.float64)
    if wife_supplemental_income > 0:
        wife_years = min(wife_years_with_supplemental_income, n_years)
        supplemental_income_per_year[:wife_years] += (
            wife_supplemental_income * inflation_factors[:wife_years]
        )
    if me_supplemental_income > 0:
        me_years = min(me_years_with_supplemental_income, n_years)
        supplemental_income_per_year[:me_years] += (
            me_supplemental_income * inflation_factors[:me_years]
        )

    # --- Run each simulation sequentially ---
    for i, start in enumerate(start_indices):
        sim_returns = returns[start : start + n_years]
        if bond_return_mode == "historical":
            assert bond_returns is not None
            sim_bond_returns = bond_returns[start : start + n_years]
        else:
            sim_bond_returns = None
        balance = float(initial_balance)
        prev_portfolio_return: Optional[float] = None

        if return_trajectories and trajectories is not None:
            trajectories[i, 0] = balance

        for t in range(n_years):
            # Year 1 has no realized prior-year return, so use regular withdrawal.
            if t == 0 or prev_portfolio_return is None:
                withdrawal_t = withdrawal
            else:
                withdrawal_t = (
                    withdrawal if prev_portfolio_return >= 0 else withdrawal_negative_year
                )
            withdrawal_t = float(withdrawal_t) * inflation_factors[t]

            # Reduce portfolio withdrawal by social security and supplemental income
            ss_amount = social_security_per_year[t]
            supp_amount = supplemental_income_per_year[t]
            withdrawal_t = max(withdrawal_t - ss_amount - supp_amount, 0.0)

            raw_equity_return = float(sim_returns[t])
            if sim_bond_returns is None:
                bond_return_t = bond_rate
            else:
                bond_return_t = float(sim_bond_returns[t])
            if hedging_config.apply_to_equity_only:
                hedged_equity_return = apply_hedge_to_equity_return(
                    raw_equity_return,
                    hedging_config,
                    periods_per_year,
                )
                portfolio_return = sp500_frac * hedged_equity_return + bond_frac * bond_return_t
            else:
                blended_return = sp500_frac * raw_equity_return + bond_frac * bond_return_t
                portfolio_return = apply_hedge_to_equity_return(
                    blended_return,
                    hedging_config,
                    periods_per_year,
                )

            portfolio_return = float(portfolio_return) - float(annual_expense_ratio)

            # Compute portfolio growth using linear allocation
            portfolio_growth = 1 + portfolio_return

            # Update balance after withdrawals and growth
            balance = (balance - withdrawal_t) * portfolio_growth

            # Reset balance to initial if it drops below initial balance in early years
            if t < go_back_year and balance < initial_balance:
                balance = initial_balance

            # Apply floor at zero
            balance = max(balance, 0.0)

            if return_trajectories and trajectories is not None:
                trajectories[i, t + 1] = balance
            prev_portfolio_return = portfolio_return

        final_balances[i] = balance

    end_time = time.time()

    print(
        f"Simulation of {n_sims:,} rolling windows ({n_years} years each) took {end_time - start_time:.2f} seconds."
    )

    # Find indices of simulations that ended below zero (only if we have trajectories)
    if return_trajectories and trajectories is not None:
        failed_indices = np.where(trajectories[:, -1] <= 0)[0]
        # Map each failed simulation index to its starting year
        date_df = pd.read_excel(file_path, sheet_name="Data", skiprows=8, usecols=[0])
        date_df.columns = ["Date"]
        date_df["Date"] = pd.to_numeric(date_df["Date"], errors="coerce")
        first_observed_year = int(np.floor(date_df["Date"].dropna().min()))
        first_year = first_observed_year + 1  # pct_change drops the first calendar year
        last_year = first_year + total_years - 1
        years = np.arange(first_year, last_year + 1)
        starting_years = years[failed_indices]
        # Print the starting year of each failed trajectory
        for y in starting_years:
            print(f"Simulation starting in {y} failed (ending balance ≤ 0).")

    overlap_note = (
        f"rolling windows overlap — fraction failed is descriptive, not i.i.d. MC "
        f"({n_sims} windows from one history)"
    )
    return SimulationData(
        initial_balance,
        withdrawal,
        withdrawal_negative_year,
        n_years,
        n_sims,
        final_balances,
        trajectories,
        total_years,
        returns,
        go_back_year=go_back_year,
        reserve_floor=reserve_floor,
        sampling_mode="historical_rolling",
        block_bootstrap_size=None,
        bond_return_mode=str(bond_return_mode),
        inflation_rate=inflation_rate,
        bond_rate=bond_rate,
        annual_expense_ratio=annual_expense_ratio,
        overlapping_windows_note=overlap_note,
    )


def inverse_exponential(value, vmin, vmax, k=5):
    """
    Maps a value between vmin and vmax to a number between 0 and 1
    with rapid decay (inverse exponential), numerically mirroring `exponential`.
    The plot looks like a logarithmic but it decays as we go towards vmax.

    Parameters:
    - value: the input value
    - vmin: minimum value (maps to 1)
    - vmax: maximum value (maps to 0)
    - k: controls steepness of the decay (default 5) 0 flat, 10 very sharp

    Returns:
    - float between 0 and 1
    """
    value = max(min(value, vmax), vmin)
    # Flip input relative to exponential
    x = vmax - (value - vmin)
    return (1 - math.exp(-k * ((x - vmin) / (vmax - vmin)))) / (1 - math.exp(-k))


def exponential(value, vmin, vmax, k=5):
    """
    Maps a value between vmin and vmax to a number between 0 and 1
    with rapid growth (exponential).
    The plot of this function looks like a logistic curve.

    Parameters:
    - value: the input value
    - vmin: minimum value (maps to 0)
    - vmax: maximum value (maps to 1)
    - k: controls steepness of the growth (default 5) 0 flat, 10 very sharp

    Returns:
    - float between 0 and 1
    """
    # Clamp value to min/max
    value = max(min(value, vmax), vmin)

    # Normalized exponential growth
    return (1 - math.exp(-k * (value - vmin) / (vmax - vmin))) / (1 - math.exp(-k))


def threshold_power_map(v: float, t: float = 0.75, k: float = 0.5) -> float:
    """
    Map v in [0,1] to:
      - 0 when v < t
      - t when v == t
      - > v (amplified) when v > t, approaching 1 at v == 1
    The plot looks like

    Parameters:
      v : input in [0,1]
      t : threshold in [0,1) (e.g. 0.75)
      k : power exponent in (0,1].  k < 1 => concave / amplifies values near 1.
            Typical: 0.3..0.6.  p==1 => linear mapping on [t,1].

    Returns:
      mapped float in [0,1]
    """
    # clamp v and t to valid ranges
    v = max(0.0, min(1.0, v))
    t = max(0.0, min(1.0, t))

    if v < t:
        return 0.0
    # normalized position within (t..1)
    s = (v - t) / (1.0 - t)  # in [0,1]
    return t + (1.0 - t) * (s**k)
