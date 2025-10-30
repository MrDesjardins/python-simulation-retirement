import time
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np


class SimulationData:
    def __init__(
        self,
        initial_balance,
        withdrawal,
        n_years,
        n_sims,
        final_balances,
        trajectories,
        total_years,
        returns_by_year,
    ):
        self.initial_balance = initial_balance
        self.withdrawal = withdrawal
        self.n_years = n_years
        self.n_sims = n_sims
        self.final_balances = final_balances
        self.trajectories = trajectories
        self.total_years = total_years
        self.returns_by_year = returns_by_year

    def print_stats(self):
        prob_success = np.mean(self.final_balances > 0)
        mean_final = np.mean(self.final_balances)
        median_final = np.median(self.final_balances)
        percentiles = np.percentile(self.final_balances, [10, 25, 75, 90])
        std_final = np.std(
            self.final_balances, ddof=1
        )  # ddof=1 for sample standard deviation
        # 95% confidence interval
        z = 1.96
        n = len(self.final_balances)
        ci_lower = mean_final - z * (std_final / np.sqrt(n))
        ci_upper = mean_final + z * (std_final / np.sqrt(n))

        # Define bin edges
        bins = [-float("inf"), -0.20, -0.10, 0, 0.10, 0.20, float("inf")]
        labels = [
            "< -20%",
            "-20% to -10%",
            "-10% to 0%",
            "0% to 10%",
            "10% to 20%",
            "20% +",
        ]
        binned = pd.cut(self.returns_by_year, bins=bins, labels=labels)

        print(f"There is {self.total_years} years of historical return data.")
        print("Return distribution over historical years:")
        print(binned.value_counts().sort_index())
        print(
            f"Probability portfolio survives {self.n_years} years: {prob_success:.1%}"
        )
        print(f"Median ending balance: ${median_final:,.0f}")
        print(
            "10th, 25th, 75th, 90th percentile outcomes:",
            [f"${p:,.0f}" for p in percentiles],
        )
        print(f"Standard deviation of ending balances: ${std_final:,.0f}")
        print(f"Mean ending balance: ${mean_final:,.0f}")
        print(
            f"95% confidence interval for the mean: ${ci_lower:,.0f} – ${ci_upper:,.0f}"
        )


# global to be set in initializer
_RETURNS = None
_N_YEARS = None
_INITIAL_BALANCE = None
_WITHDRAWAL = None


def _init_worker(returns, n_years, initial_balance, withdrawal):
    global _RETURNS, _N_YEARS, _INITIAL_BALANCE, _WITHDRAWAL
    _RETURNS = returns
    _N_YEARS = n_years
    _INITIAL_BALANCE = initial_balance
    _WITHDRAWAL = withdrawal


def _simulate_chunk(args):
    """
    Worker: simulate `chunk_size` independent simulations.
    Returns final_balances and (optionally) trajectories.
    args: (chunk_size, seed, return_trajectories_bool)
    """
    chunk_size, seed, return_traj = args
    rng = np.random.default_rng(seed)
    m = len(_RETURNS)
    # Generate shuffled indices: shape (chunk_size, n_years)
    # memory: chunk_size * n_years * 8 bytes
    idx = np.empty((chunk_size, _N_YEARS), dtype=np.int64)
    for i in range(chunk_size):
        idx[i] = rng.permutation(m)[:_N_YEARS]

    sim_returns = _RETURNS[idx]  # shape (chunk_size, n_years)
    balances = np.full(chunk_size, _INITIAL_BALANCE, dtype=np.float64)

    if return_traj:
        trajectories = np.zeros((chunk_size, _N_YEARS + 1), dtype=np.float64)
        trajectories[:, 0] = _INITIAL_BALANCE
    else:
        trajectories = None

    inflation_rate = 0.03
    # Vectorized year loop (fast: operates on arrays of size chunk_size)
    for t in range(_N_YEARS):
        balances = (balances - _WITHDRAWAL * ((1 + inflation_rate) ** t)) * (
            1.0 + sim_returns[:, t]
        )
        # floor at zero:
        np.maximum(balances, 0.0, out=balances)
        if return_traj:
            trajectories[:, t + 1] = balances

    return balances, trajectories


def run_simulation_mp(
    n_sims=1_000_000,
    n_years=45,
    initial_balance=6_000_000,
    withdrawal=70_000,
    n_workers=None,
    return_trajectories=False,
    chunk_size=None,
):
    # Load returns from your file (same as before)
    start_time = time.time()
    file_path = "data/ie_data.xls"
    df = pd.read_excel(file_path, sheet_name="Data", skiprows=8)
    df = df.iloc[:, [0, 9]]
    df.columns = ["Date", "Real Total Return Price"]
    df = df.dropna()
    df["Year"] = df["Date"].astype(str).str.split(".").str[0].astype(int)
    annual = df.groupby("Year")["Real Total Return Price"].last().dropna()
    returns = annual.pct_change().dropna().to_numpy()
    total_years = len(returns)
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # leave one core for OS/other tasks

    if chunk_size is None:
        # choose chunk_size so each worker gets a few thousand sims
        # avoid too many small tasks; tune for your memory
        chunk_size = max(1_000, n_sims // (n_workers * 4))

    # Create the task list of (chunk_size, seed, return_trajectories)
    tasks = []
    remaining = n_sims
    seed_base = np.random.SeedSequence().entropy  # base entropy
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
        initargs=(returns, n_years, initial_balance, withdrawal),
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
    if return_trajectories:
        trajectories = trajectories[:n_sims]

    end_time = time.time()
    print(
        f"Simulation of {n_sims:,} runs over {n_years} years took {end_time - start_time:.2f} seconds."
    )
    # Return a simple object (adapt to your SimulationData)
    return SimulationData(
        initial_balance,
        withdrawal,
        n_years,
        n_sims,
        final_balances,
        trajectories,
        total_years,
        returns_by_year=returns,
    )
