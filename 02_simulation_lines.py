import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from common import run_simulation_mp

simulation_data = run_simulation_mp(
    n_years=40,
    return_trajectories=True,
    n_sims=1_000_000,
    initial_balance=4_600_000,
    withdrawal=100_000,
    withdrawal_negative_year=100_000,
    random_with_real_life_constraints=True,
    sp500_percentage=0.65,
    bond_rate=0.03,
    inflation_rate=0.03,
    social_security_money=50_000,
    years_without_social_security=20,
    years_with_supplemental_income=12,
    supplemental_income=30_000,
)
simulation_data.print_stats()
trajectories = simulation_data.trajectories
n_sims = simulation_data.n_sims
n_years = simulation_data.n_years
plt.figure(figsize=(20, 10))

# Choose only random simulations to plot for clarity
size = min(50_000, n_sims)
selected_indices = np.random.choice(n_sims, size=size, replace=False)
for sim in selected_indices:
    # Red if ending balance <= 0, otherwise blue
    color = "red" if trajectories[sim, -1] <= 0 else "blue"
    alpha = 0.1 if color == "red" else 0.01
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
