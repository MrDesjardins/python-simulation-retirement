import os
import time
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from common import SimulationData, format_withdrawal_breakdown, run_simulation_mp
from simulation_scenario import BASELINE_SUCCESS_SCRIPT, format_config_lines

# Sampling mode for historical returns (overrides baseline if changed):
#   "block_bootstrap" (default, recommended) — preserves multi-year crash regimes
#   "random"                                  — single-year, no autocorrelation
#   "constrained"                             — bounded streak constraints
SAMPLING_MODE = "block_bootstrap"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _required_sims_for_half_width(
    probability: float, half_width: float, z: float = 1.96
) -> int:
    """Approximate simulations needed for a confidence-interval half-width."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if half_width <= 0.0:
        raise ValueError(f"half_width must be > 0, got {half_width}")
    # At an observed rate of exactly 0% or 100%, the plug-in variance is zero,
    # but that does not mean the underlying probability is known exactly. Use
    # the conservative Bernoulli variance maximum for planning in that case.
    variance = probability * (1.0 - probability)
    if variance == 0.0:
        variance = 0.25
    return max(1, ceil(variance * (z / half_width) ** 2))


def _print_simulation_coverage(
    simulation_data: SimulationData,
    *,
    n_sims: int,
    sampling_mode: str,
    block_size: int,
) -> None:
    """Print Monte Carlo precision and historical-scenario coverage guidance."""
    probability = simulation_data.probability_of_success
    half_width_95 = (
        simulation_data.wilson_success_95[1] - simulation_data.wilson_success_95[0]
    ) / 2
    n_years = simulation_data.n_years
    total_years = simulation_data.total_years
    print("\n--- Simulation coverage / precision ---")
    print(
        f"  Monte Carlo paths: {n_sims:,} "
        f"({n_sims * n_years:,} simulated retirement-years)"
    )
    print(f"  Historical return years available: {total_years:,}")
    print(f"  Horizon: {n_years} years; sampling mode: {sampling_mode}")
    if sampling_mode == "block_bootstrap":
        blocks_per_path = ceil(n_years / block_size)
        print(
            f"  Block bootstrap: {block_size}-year blocks "
            f"({blocks_per_path} blocks/path)"
        )
    print(
        f"  Observed outcomes: {simulation_data.success_count:,} successes, "
        f"{n_sims - simulation_data.success_count:,} failures"
    )
    print(
        f"  95% uncertainty around P(success): "
        f"±{half_width_95 * 100:.4f} percentage points"
    )
    print("\n  Monte Carlo precision check:")
    for target in (0.01, 0.005, 0.001):
        required = _required_sims_for_half_width(probability, target)
        comparison = "enough" if n_sims >= required else "increase needed"
        print(
            f"    For ±{target * 100:.1f} percentage points (95%): "
            f"about {required:,} paths ({comparison})"
        )

    finest_target = 0.001
    finest_required = _required_sims_for_half_width(probability, finest_target)
    if n_sims >= finest_required:
        print(
            f"  Recommendation: do NOT increase N_SIMS for numerical precision. "
            f"Your {n_sims:,} paths exceed the estimate for ±{finest_target * 100:.1f} "
            "percentage-point precision."
        )
    else:
        print(
            f"  Recommendation: increase N_SIMS if you want ±{finest_target * 100:.1f} "
            f"percentage-point precision; the estimate is {finest_required:,} paths."
        )

    expected_failures = n_sims * (1.0 - probability)
    if expected_failures < 100:
        print(
            f"  Warning: only about {expected_failures:.1f} failures are represented; "
            "tail estimates may be unstable even with a small CI."
        )
    print(
        "  Important: this is not a percentage of all possible futures. N_SIMS only "
        "reduces random measurement noise within this model. It cannot tell us that "
        "98% of real-world possibilities have been covered. Scenario uncertainty "
        "comes from the historical data, block size, inflation assumptions, and "
        "other model choices."
    )


def _print_plain_language_result(simulation_data: SimulationData) -> None:
    """Print the decision metrics without relying on retirement jargon."""
    print("\n--- What does this result mean? ---")
    print(
        "Success means the portfolio still has money at the end of the "
        f"{simulation_data.n_years}-year retirement. It does not mean the "
        "balance stayed comfortably high."
    )
    print(
        f"  Portfolio still had money at the end: "
        f"{simulation_data.probability_of_success:.2%}"
    )
    print(
        "  Portfolio ran out of money before the end: "
        f"{1.0 - simulation_data.probability_of_success:.2%}"
    )

    floor = simulation_data.reserve_floor
    ever_below_floor = simulation_data.probability_ever_below_reserve_floor
    if floor is None or ever_below_floor is None:
        print("  Minimum-balance safety check: not configured")
        return

    print(
        f"  Reserve floor: ${floor:,.0f} minimum balance target "
        "(a safety cushion, not an additional withdrawal)"
    )
    print(
        f"  Fell below that safety cushion at any time: {ever_below_floor:.2%}"
    )
    print(
        f"  Stayed at or above that safety cushion throughout: "
        f"{1.0 - ever_below_floor:.2%}"
    )
    print(
        "  Use the last metric for a more conservative interpretation of "
        "retirement safety; use the first metric for simple portfolio survival."
    )


def main() -> None:
    cfg = BASELINE_SUCCESS_SCRIPT
    n_sims = _env_int("N_SIMS", 5_000_000)
    cap = _env_int("HIST_CAP", 100_000_000)
    bin_width = _env_int("HIST_BIN_WIDTH", 1_000_000)
    show_plot = _env_flag("SHOW_PLOT", True)

    base_kw = cfg.run_simulation_mp_kwargs()
    base_kw["sampling_mode"] = SAMPLING_MODE

    for line in format_config_lines("Scenario (simulation_scenario + SAMPLING_MODE):", base_kw):
        print(line)

    start_time = time.perf_counter()
    simulation_data = run_simulation_mp(
        n_sims=n_sims,
        return_trajectories=False,
        **base_kw,
    )
    end_time = time.perf_counter()
    print(
        "Run config: "
        f"n_sims={n_sims:,}, sampling_mode={SAMPLING_MODE}, "
        f"bond_return_mode={cfg.bond_return_mode}, "
        "withdrawal_switch=prior_year_portfolio_return, "
        "timestep=annual"
    )
    print(
        "Model note: annual timestep approximation (no intra-year cashflow/price path modeling)."
    )
    for line in format_withdrawal_breakdown(**cfg.format_withdrawal_breakdown_kwargs()):
        print(line)
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")
    simulation_data.print_stats()
    _print_plain_language_result(simulation_data)

    final_balances = simulation_data.final_balances
    n_sims_run = simulation_data.n_sims
    n_years = simulation_data.n_years

    _print_simulation_coverage(
        simulation_data,
        n_sims=n_sims_run,
        sampling_mode=SAMPLING_MODE,
        block_size=cfg.block_bootstrap_size,
    )

    underflow = final_balances[final_balances <= 0]
    normal = final_balances[(final_balances > 0) & (final_balances <= cap)]
    overflow = final_balances[final_balances > cap]

    bins = np.arange(0, cap + bin_width, bin_width, dtype=float).tolist()

    plt.figure(figsize=(16, 8))

    plt.hist(normal, bins=bins, edgecolor="black", label="Ending balance > $0")

    if len(underflow) > 0:
        plt.bar(
            -bin_width / 2,
            len(underflow),
            width=bin_width,
            color="gray",
            edgecolor="black",
            label="<= $0",
        )

    plt.title(f"Monte Carlo Results ({n_sims_run:,} runs, {n_years} years)")
    plt.xlabel("Ending Balance ($)")
    plt.ylabel("Frequency")

    jump = 4
    tick_positions = list(bins[::jump])
    tick_positions = [-bin_width / 2] + tick_positions
    tick_labels = ["<= $0"] + [
        f"${int(x / 1_000_000)}M" if x >= 1_000_000 else f"${x:,.0f}"
        for x in bins[::jump]
    ]
    plt.xticks(tick_positions, tick_labels, rotation=30, ha="right", fontsize=10)

    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None, numticks=10))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    if len(overflow) > 0:
        plt.figtext(
            0.99,
            0.01,
            f">{int(cap / 1_000_000)}M outcomes: {len(overflow):,}",
            ha="right",
            va="bottom",
            fontsize=10,
            color="red",
        )
    plt.legend()
    plt.tight_layout()

    # In headless sessions, save instead of trying to open a window.
    if show_plot and os.getenv("DISPLAY"):
        plt.show()
    else:
        output_path = Path("results/01_success_hist.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved histogram: {output_path}")


if __name__ == "__main__":
    main()
