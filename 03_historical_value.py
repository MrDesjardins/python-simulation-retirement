import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from common import run_simulation_historical_real
from simulation_scenario import BASELINE_SUCCESS_SCRIPT, format_config_lines


def main() -> None:
    cfg = BASELINE_SUCCESS_SCRIPT
    hist_kw = {**cfg.run_simulation_historical_real_kwargs(), "return_trajectories": True}
    for line in format_config_lines("03_historical_value (aligned with baseline scenario):", hist_kw):
        print(line)

    simulation_data = run_simulation_historical_real(**hist_kw)
    simulation_data.print_stats()
    trajectories = simulation_data.trajectories
    assert trajectories is not None
    n_sims = simulation_data.n_sims
    n_years = simulation_data.n_years
    plt.figure(figsize=(20, 10))

    for sim in range(n_sims):
        color = "red" if trajectories[sim, -1] <= 0 else "blue"
        alpha = 0.5 if color == "red" else 0.1
        plt.plot(range(n_years + 1), trajectories[sim], color=color, alpha=alpha)

    median_trajectory = np.median(trajectories, axis=0)
    plt.plot(
        range(n_years + 1), median_trajectory, color="black", linewidth=2, label="Median"
    )
    plt.legend()

    p10 = np.percentile(trajectories, 10, axis=0)
    p90 = np.percentile(trajectories, 90, axis=0)
    plt.fill_between(
        range(n_years + 1), p10, p90, color="gray", alpha=0.3, label="10–90th percentile"
    )

    plt.xlabel("Year")
    plt.ylabel("Portfolio Balance ($)")
    plt.title(f"Portfolio Trajectories Over {n_years} Years ({n_sims:,} simulations)")
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if os.getenv("DISPLAY"):
        plt.show()
    else:
        output_path = Path("results/03_historical_value.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
