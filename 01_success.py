import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from common import format_withdrawal_breakdown, run_simulation_mp

# Sampling mode for historical returns:
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


def main() -> None:
    # Keep this configurable: override with N_SIMS=... when needed.
    # Example: N_SIMS=5000000 uv run 01_success.py
    n_sims = _env_int("N_SIMS", 2_000_000)
    cap = _env_int("HIST_CAP", 160_000_000)
    bin_width = _env_int("HIST_BIN_WIDTH", 1_000_000)
    show_plot = _env_flag("SHOW_PLOT", True)
    withdrawal = 120_000  # 120k is 96k after taxes
    withdrawal_negative_year = 100_000  # 100k is 80k after taxes
    social_security_money = 48_000
    years_without_social_security = 25
    years_with_supplemental_income = 12
    supplemental_income = 30_000

    start_time = time.perf_counter()
    simulation_data = run_simulation_mp(
        n_years=40,
        return_trajectories=False,
        n_sims=n_sims,
        initial_balance=4_300_000, # Does not containt the 500k for house down payment
        sampling_mode=SAMPLING_MODE,
        withdrawal=withdrawal,
        withdrawal_negative_year=withdrawal_negative_year,
        random_with_real_life_constraints=False,
        sp500_percentage=0.75,
        bond_rate=0.03,  # Only if bond_return_mode is "fixed"
        bond_return_mode="historical",
        inflation_rate=0.03,
        social_security_money=social_security_money,
        years_without_social_security=years_without_social_security,
        years_with_supplemental_income=years_with_supplemental_income,
        supplemental_income=supplemental_income,
    )
    end_time = time.perf_counter()
    print(
        "Run config: "
        f"n_sims={n_sims:,}, sampling_mode={SAMPLING_MODE}, "
        "bond_return_mode=historical, "
        "withdrawal_switch=prior_year_portfolio_return, "
        "timestep=annual"
    )
    print(
        "Model note: annual timestep approximation (no intra-year cashflow/price path modeling)."
    )
    for line in format_withdrawal_breakdown(
        withdrawal=withdrawal,
        withdrawal_negative_year=withdrawal_negative_year,
        supplemental_income=supplemental_income,
        years_with_supplemental_income=years_with_supplemental_income,
        social_security_money=social_security_money,
        years_without_social_security=years_without_social_security,
        n_years=40,
    ):
        print(line)
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")
    simulation_data.print_stats()

    final_balances = simulation_data.final_balances
    n_sims_run = simulation_data.n_sims
    n_years = simulation_data.n_years

    underflow = final_balances[final_balances <= 0]
    normal = final_balances[(final_balances > 0) & (final_balances <= cap)]
    overflow = final_balances[final_balances > cap]

    bins = np.arange(0, cap + bin_width, bin_width, dtype=float).tolist()

    plt.figure(figsize=(16, 8))

    plt.hist(normal, bins=bins, edgecolor="black")

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
    tick_labels = (
        ["<= $0"]
        + [
            f"${int(x / 1_000_000)}M" if x >= 1_000_000 else f"${x:,.0f}"
            for x in bins[::jump]
        ]
    )
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
