import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from common import run_simulation_historical_real

simulation_data = run_simulation_historical_real(return_trajectories=True)
simulation_data.print_stats()
trajectories = simulation_data.trajectories
n_sims = simulation_data.n_sims
n_years = simulation_data.n_years
plt.figure(figsize=(20, 10))


for sim in range(n_sims):
    color = "red" if trajectories[sim, -1] <= 0 else "blue"
    alpha = 0.5 if color == "red" else 0.2
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
plt.show()
