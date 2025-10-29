import pandas as pd
import numpy as np


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
    n_sims=50_000, n_years=40, initial_balance=5_000_000, withdrawal=100_000
):
    """
    Load the data from the Excel file and run the simulations.
    """
    file_path = "data/ie_data.xls"
    # Skip the header
    df = pd.read_excel(file_path, sheet_name="Data", skiprows=8)

    # We'll keep Date (col 0) and column Real Total Return Price (col 9)
    df = df.iloc[:, [0, 9]]
    df.columns = ["Date", "Real Total Return Price"]  # Rename for clarity
    df = df.dropna()
    df["Year"] = df["Date"].astype(str).str.split(".").str[0].astype(int)
    # Take the last (december)
    annual = df.groupby("Year")["Real Total Return Price"].last().dropna()
    returns = annual.pct_change().dropna().to_numpy()

    final_balances = []
    trajectories = np.zeros((n_sims, n_years + 1))
    trajectories[:, 0] = initial_balance  # starting balance

    for i in range(n_sims):
        balance = initial_balance
        shuffled_returns = np.random.permutation(returns)
        for t, r in enumerate(shuffled_returns[:n_years]):
            balance = (balance - withdrawal) * (1 + r)
            if balance <= 0:
                balance = 0
            trajectories[i, t + 1] = balance
        final_balances.append(balance)

    final_balances = np.array(final_balances)
    return SimulationData(
        initial_balance, withdrawal, n_years, n_sims, final_balances, trajectories
    )
