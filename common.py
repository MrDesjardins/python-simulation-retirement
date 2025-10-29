import pandas as pd
import numpy as np
import time

class SimulationData:
    def __init__(
        self, initial_balance, withdrawal, n_years, n_sims, final_balances, trajectories
    ):
        self.initial_balance = initial_balance
        self.withdrawal = withdrawal
        self.n_years = n_years
        self.n_sims = n_sims
        self.final_balances = final_balances
        self.trajectories = trajectories

    def print_stats(self):
        prob_success = np.mean(self.final_balances > 0)
        mean_final = np.mean(self.final_balances)
        median_final = np.median(self.final_balances)
        percentiles = np.percentile(self.final_balances, [10, 25, 75, 90])
        std_final = np.std(self.final_balances, ddof=1)  # ddof=1 for sample standard deviation
        # 95% confidence interval
        z = 1.96
        n = len(self.final_balances)
        ci_lower = mean_final - z * (std_final / np.sqrt(n))
        ci_upper = mean_final + z * (std_final / np.sqrt(n))

        print(f"Probability portfolio survives {self.n_years} years: {prob_success:.1%}")
        print(f"Median ending balance: ${median_final:,.0f}")
        print("10th, 25th, 75th, 90th percentile outcomes:", [f"${p:,.0f}" for p in percentiles])
        print(f"Standard deviation of ending balances: ${std_final:,.0f}")
        print(f"Mean ending balance: ${mean_final:,.0f}")
        print(f"95% confidence interval for the mean: ${ci_lower:,.0f} – ${ci_upper:,.0f}")


def run_simulation(
    n_sims=1_000_000, n_years=25, initial_balance=2_000_000, withdrawal=50_000
):
    """
    Load the data from the Excel file and run the simulations.
    """
    start_time = time.time()
    file_path = "data/ie_data.xls"
    df = pd.read_excel(file_path, sheet_name="Data", skiprows=8)
    df = df.iloc[:, [0, 9]]
    df.columns = ["Date", "Real Total Return Price"]
    df = df.dropna()
    df["Year"] = df["Date"].astype(str).str.split(".").str[0].astype(int)
    annual = df.groupby("Year")["Real Total Return Price"].last().dropna()
    returns = annual.pct_change().dropna().to_numpy()

    # Pre-generate all shuffled returns for all simulations
    idx = np.arange(len(returns))
    shuffled_idx = np.array([np.random.permutation(idx)[:n_years] for _ in range(n_sims)])
    sim_returns = returns[shuffled_idx]  # shape: (n_sims, n_years)

    # Initialize trajectories
    trajectories = np.zeros((n_sims, n_years + 1))
    trajectories[:, 0] = initial_balance

    # Vectorized simulation
    balances = np.full(n_sims, initial_balance, dtype=np.float64)
    for t in range(n_years):
        balances = (balances - withdrawal) * (1 + sim_returns[:, t])
        balances = np.maximum(balances, 0)  # floor at 0
        trajectories[:, t + 1] = balances
    end_time = time.time()
    print(f"Simulation of {n_sims} runs over {n_years} years took {end_time - start_time:.2f} seconds.")
    return SimulationData(
        initial_balance, withdrawal, n_years, n_sims, balances, trajectories
    )
