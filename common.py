import time
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from random_utils import generate_constrained_indices


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
        self.probability_of_success = np.mean(final_balances > 0)
        self.std_final = np.std(final_balances, ddof=1) # ddof=1 for sample standard deviation
        self.std_error = self.std_final / np.sqrt(n_sims)

    def print_stats(self):
        prob_success = self.probability_of_success
        mean_final = np.mean(self.final_balances)
        median_final = np.median(self.final_balances)
        percentiles = np.percentile(self.final_balances, [10, 25, 75, 90])
        std_final = self.std_final
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
        print(
            f"The simulation starts using a portfolio balance of ${self.initial_balance:,.0f}."
        )
        print(f"There is {self.total_years} years of historical return data.")
        print("Return distribution over historical years:")
        print(binned.value_counts().sort_index())
        print(
            f"\nProbability portfolio survives {self.n_years} years: {prob_success:.1%} if withdrawing ${self.withdrawal:,.0f} per year and in negative years ${self.withdrawal_negative_year:,.0f} per year."
        )

        # Separate the three categories
        underflow = self.final_balances[self.final_balances <= 0]
        overflow = self.final_balances[self.final_balances > 0]
        print(f"{len(underflow):,} simulations ended ≤ $0.")
        print(f"{len(overflow):,} simulations ended > $0.")

        print(f"\nMedian ending balance: ${median_final:,.0f}")
        print(
            "10th, 25th, 75th, 90th percentile outcomes:",
            [f"${p:,.0f}" for p in percentiles],
        )
        print(f"Standard deviation of ending balances: ${std_final:,.0f}")
        print(f"Mean ending balance: ${mean_final:,.0f} with standard error ${self.std_error:,.0f}")
        print(
            f"95% confidence interval for the mean: ${ci_lower:,.0f} – ${ci_upper:,.0f}"
        )
        # Compute year-over-year changes`
        if self.trajectories is not None:
            np.set_printoptions(suppress=True, precision=2, linewidth=150)

            yearly_change = self.trajectories[:, 1:] - self.trajectories[:, :-1]

            #print(yearly_change)
            # Boolean array: True if negative change (loss)
            loss_years = yearly_change < 0
            # Sliding window of length along years
            window_size = 5
            windows = np.lib.stride_tricks.sliding_window_view(
                loss_years, window_size, axis=1
            )

            consecutive_year_lost = np.all(
                windows, axis=2
            )  # shape: (n_sims, n_years - 3)

            sim_with_year_neg = np.any(consecutive_year_lost, axis=1)

            print(
                f"{np.sum(sim_with_year_neg)} simulations ({np.sum(sim_with_year_neg) / len(sim_with_year_neg):.1%}) had {window_size} or more consecutive negative years."
            )
            frac_loss = np.sum(loss_years, axis=1) / loss_years.shape[1]

            sim_more_than_50pct_loss = frac_loss > 0.5

            print(
                f"{np.sum(sim_more_than_50pct_loss)} simulations ({np.sum(sim_more_than_50pct_loss) / len(sim_more_than_50pct_loss):.1%}) had more than 50% negative years."
            )


# global to be set in initializer
_RETURNS = None
_N_YEARS = None
_INITIAL_BALANCE = None
_WITHDRAWAL = None
_WITHDRAWAL_NEGATIVE_YEAR = None
_GO_BACK_YEARS = None
_RANDOM_CONSTRAINED_INDICES = None


def _init_worker(
    returns,
    n_years,
    initial_balance,
    withdrawal,
    withdrawal_negative_year,
    go_back_year,
    random_with_real_life_constraints,
):
    global _RETURNS, _N_YEARS, _INITIAL_BALANCE, _WITHDRAWAL, _WITHDRAWAL_NEGATIVE_YEAR, _GO_BACK_YEARS, _RANDOM_CONSTRAINED_INDICES
    _RETURNS = returns
    _N_YEARS = n_years
    _INITIAL_BALANCE = initial_balance
    _WITHDRAWAL = withdrawal
    _WITHDRAWAL_NEGATIVE_YEAR = withdrawal_negative_year
    _GO_BACK_YEARS = go_back_year
    _RANDOM_CONSTRAINED_INDICES = random_with_real_life_constraints


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
    if _RANDOM_CONSTRAINED_INDICES:
        for i in range(chunk_size):
            idx[i] = generate_constrained_indices(
                rng, _RETURNS, _N_YEARS, max_consec_neg=4, max_consec_drop=-0.5
            )
    else:
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
        withdrawals = np.where(
            sim_returns[:, t] >= 0, _WITHDRAWAL, _WITHDRAWAL_NEGATIVE_YEAR
        )
        # Apply inflation
        withdrawals = withdrawals * ((1 + inflation_rate) ** t)

        # Update balances
        balances = (balances - withdrawals) * (1.0 + sim_returns[:, t])

        # In the case the balance goes under the initial money in the first 5 years, we
        # go back to work to get back to the initial balance
        if t < _GO_BACK_YEARS:
            need_to_work_scenario = balances < _INITIAL_BALANCE
            balances[need_to_work_scenario] = _INITIAL_BALANCE

        # Apply a floor at zero:
        np.maximum(balances, 0.0, out=balances)
        if return_traj:
            trajectories[:, t + 1] = balances

    return balances, trajectories


def run_simulation_mp(
    n_sims=100_000,
    n_years=45,
    initial_balance=6_000_000,
    withdrawal=120_000,
    withdrawal_negative_year=40_000,
    go_back_year=0,
    n_workers=None,
    return_trajectories=False,
    chunk_size=None,
    random_with_real_life_constraints=True,
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
        initargs=(
            returns,
            n_years,
            initial_balance,
            withdrawal,
            withdrawal_negative_year,
            go_back_year,
            random_with_real_life_constraints,
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
    if return_trajectories:
        trajectories = trajectories[:n_sims]

    end_time = time.time()
    mode = (
        "constrained random"
        if random_with_real_life_constraints
        else "unconstrained random"
    )
    print(
        f"Simulation of {n_sims:,} runs over {n_years} years took {end_time - start_time:.2f} seconds using {mode}."
    )

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
        returns_by_year=returns,
    )


def run_simulation_historical_real(
    n_years=45,
    initial_balance=4_800_000,
    withdrawal=120_000,
    withdrawal_negative_year=80_000,
    go_back_year=0,
    return_trajectories=False,
):
    start_time = time.time()

    # --- Load and prepare historical data ---
    file_path = "data/ie_data.xls"
    df = pd.read_excel(file_path, sheet_name="Data", skiprows=8)
    df = df.iloc[:, [0, 9]]
    df.columns = ["Date", "Real Total Return Price"]
    df = df.dropna()
    df["Year"] = df["Date"].astype(str).str.split(".").str[0].astype(int)

    annual = df.groupby("Year")["Real Total Return Price"].last().dropna()
    returns = annual.pct_change().dropna().to_numpy()
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

    inflation_rate = 0.03

    # --- Run each simulation sequentially ---
    for i, start in enumerate(start_indices):
        sim_returns = returns[start : start + n_years]
        balance = float(initial_balance)

        if return_trajectories:
            trajectories[i, 0] = balance

        for t in range(n_years):
            withdrawal_t = (
                withdrawal if sim_returns[t] >= 0 else withdrawal_negative_year
            )
            withdrawal_t = float(withdrawal_t) * ((1 + inflation_rate) ** t)
            balance = (balance - withdrawal_t) * (1.0 + sim_returns[t])

            if t < go_back_year and balance < initial_balance:
                balance = initial_balance

            balance = max(balance, 0.0)

            if return_trajectories:
                trajectories[i, t + 1] = balance

        final_balances[i] = balance

    end_time = time.time()

    print(
        f"Simulation of {n_sims:,} rolling windows ({n_years} years each) took {end_time - start_time:.2f} seconds."
    )

    # Find indices of simulations that ended below zero
    failed_indices = np.where(trajectories[:, -1] <= 0)[0]
    # Map each failed simulation index to its starting year
    first_year = annual.index[0]
    last_year = annual.index[-1] - n_years + 1
    years = np.arange(first_year, last_year + 1)
    starting_years = years[failed_indices]
    # Print the starting year of each failed trajectory
    for y in starting_years:
        print(f"Simulation starting in {y} failed (ending balance ≤ 0).")

    return SimulationData(
        initial_balance,
        withdrawal,
        withdrawal_negative_year,
        n_years,
        n_sims,
        final_balances,
        trajectories,
        total_years,
        returns_by_year=returns,
    )
