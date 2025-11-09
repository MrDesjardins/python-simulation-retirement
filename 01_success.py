import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from common import run_simulation_mp

# simulation_data = run_simulation()
simulation_data = run_simulation_mp(
    n_years=40,
    return_trajectories=False,
    n_sims=100_000,
    initial_balance=4_500_000,
    withdrawal=110_000,
    withdrawal_negative_year=95_000,
    random_with_real_life_constraints=True,
    sp500_percentage=0.1,
    bond_rate=0.035
)
simulation_data.print_stats()
final_balances = simulation_data.final_balances
n_sims = simulation_data.n_sims
n_years = simulation_data.n_years

cap = 60_000_000
bin_width = 1_000_000

# Separate the three categories
underflow = final_balances[final_balances <= 0]
normal = final_balances[(final_balances > 0) & (final_balances <= cap)]
overflow = final_balances[final_balances > cap]

# Define bins up to $40M
bins = np.arange(0, cap + bin_width, bin_width)

plt.figure(figsize=(12, 8))

# Plot normal balances
counts, edges, patches = plt.hist(normal, bins=bins, edgecolor="black")

if len(underflow) > 0:
    plt.bar(
        -bin_width / 2,
        len(underflow),
        width=bin_width,
        color="gray",
        edgecolor="black",
        label="≤ $0",
    )

if len(overflow) > 0:
    plt.bar(
        cap + bin_width / 2,
        len(overflow),
        width=bin_width,
        color="red",
        edgecolor="black",
        label="40M+",
    )

plt.title(f"Monte Carlo Results ({n_sims:,} runs, {n_years} years)")
plt.xlabel("Ending Balance ($)")
plt.ylabel("Frequency")
jump = 4
tick_positions = list(bins[::jump])  # every 1M for readability
tick_positions = [-bin_width / 2] + tick_positions + [cap + bin_width / 2]
tick_labels = (
    ["≤ $0"]
    + [
        f"${int(x/1_000_000)}M" if x >= 1_000_000 else f"${x:,.0f}"
        for x in bins[::jump]
    ]
    + ["40M+"]
)
plt.xticks(tick_positions, tick_labels, rotation=30, ha="right", fontsize=10)

plt.yscale("log")
ax = plt.gca()
ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None, numticks=10))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
plt.legend()
plt.tight_layout()
plt.show()
