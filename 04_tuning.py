import optuna
import numpy as np
from common import (
    exponential,
    inverse_exponential,
    run_simulation_mp,
    threshold_power_map,
)

# Thresholds for adaptive simulation
STD_ERROR_DIFF_THRESHOLD = 0.02  # stop increasing n_sims if std_error stabilizes
STD_ERROR_ACCEPTANCE = 0.005  # accept std_error if small enough

# Constants
INITIAL_BALANCE_RANGE = (3_000_000, 6_000_000)
INITIAL_BALANCE_STEP = 250_000
WITHDRAWAL_RANGE = (100_000, 140_000)
WITHDRAWAL_STEP = 5_000
WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE = (0.85, 1.0)
WITHDRAWAL_NEGATIVE_STEP = 0.05
N_SIMS_RANGE = (25_000, 500_000)
STEP_N_SIMS = 25_000
TRIAL_COUNT = 150
STORAGE_PATH = "sqlite:///db.sqlite3"
STUDY_NAME = (
    "retirement_tuning_study_v47"  # ⚠️ CHANGE EVERYTIME WE CHANGE CONSTANTS OR LOGIC ⚠️
)
REAL_LIFE_CONSTRAINTS = True
RETIREMENT_YEARS = 40
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND = 0.5
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP = 0.05
INFLATION_RATE = 0.03
BOND_RATE = 0.03
MAX_GAIN_MULTIPLIER = 2.0


def objective(trial):
    # Suggest both initial balance and withdrawal amount
    initial_balance = trial.suggest_float(
        "initial_balance", *INITIAL_BALANCE_RANGE, step=INITIAL_BALANCE_STEP
    )
    withdrawal = trial.suggest_float(
        "withdrawal", *WITHDRAWAL_RANGE, step=WITHDRAWAL_STEP
    )
    withdrawal_percentage_when_negative_year = trial.suggest_float(
        "withdrawal_percentage_when_negative_year",
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[0],
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[1],
        step=WITHDRAWAL_NEGATIVE_STEP,
    )
    withdrawal_negative_year = trial.suggest_float(
        "withdrawal_negative_year",
        withdrawal * withdrawal_percentage_when_negative_year,
        withdrawal,
        step=WITHDRAWAL_STEP,
    )
    sp500_percentage = trial.suggest_float(
        "sp500_percentage",
        PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND,
        1,
        step=PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP,
    )

    # Adaptive Monte Carlo configuration
    n_sims = N_SIMS_RANGE[0]
    max_n_sims = N_SIMS_RANGE[1]
    step_n_sims = STEP_N_SIMS
    last_std_error = float("inf")

    while n_sims <= max_n_sims:
        simulation_data = run_simulation_mp(
            n_sims=n_sims,
            initial_balance=initial_balance,
            random_with_real_life_constraints=REAL_LIFE_CONSTRAINTS,
            withdrawal=withdrawal,
            withdrawal_negative_year=withdrawal_negative_year,
            n_years=RETIREMENT_YEARS,
            sp500_percentage=sp500_percentage,
            bond_rate=BOND_RATE,
            inflation_rate=INFLATION_RATE,
        )

        std_error = simulation_data.std_error
        diff_compare_last = (
            (last_std_error - std_error) / last_std_error
            if last_std_error != float("inf")
            else float("inf")
        )

        std_error_relative_to_mean = std_error / simulation_data.final_balances.mean()

        # print(
        #     f"Trial {trial.number} | n_sims={n_sims:,} | std_error={std_error:.4f} | final_balance_mean={simulation_data.final_balances.mean():.4f} | "
        #     f"diff_compare_last={diff_compare_last:.4%} | std_error_rel={std_error_relative_to_mean:.4%}"
        # )

        # Stop if we have good convergence
        if std_error_relative_to_mean <= STD_ERROR_ACCEPTANCE or (
            diff_compare_last != float("inf")
            and diff_compare_last <= STD_ERROR_DIFF_THRESHOLD
        ):
            break

        last_std_error = std_error
        n_sims += step_n_sims

    # Store how many simulations were used for this trial
    trial.set_user_attr("n_sims_used", n_sims)

    # Already between 0 and 1
    prob_success = simulation_data.probability_of_success

    # Composite scoring logic (should be between 0 and 1 but does not have to)
    #   1. Prioritize success rate
    #   2. Among high success, prefer higher withdrawal
    #   3. Among equal withdrawal, prefer lower initial balance
    #   Each term is scaled to similar magnitude.

    final_balances = simulation_data.final_balances
    if final_balances.ndim == 2:
        last_year_balances = final_balances[:, -1]
    else:
        last_year_balances = final_balances
    final_balance_growth_ratios = last_year_balances / initial_balance
    final_balance_relative_gain = np.clip(
        (final_balance_growth_ratios - 1) / (2 - 1), 0, 1
    )  # 2x cap
    final_balance_average_relative_median = np.median(final_balance_relative_gain)
    final_balance_average_relative_min = np.min(final_balance_relative_gain)
    final_balance_average_relative_max = np.max(final_balance_relative_gain)
    # gap_ratio = np.clip((withdrawal - withdrawal_negative_year) / withdrawal, 0, 1)
    score = (
        threshold_power_map(prob_success, 0.85, 0.2) * 0.65
        + exponential(withdrawal, *WITHDRAWAL_RANGE, 5)
        * 0.05  # Encourage higher withdrawals
        + inverse_exponential(initial_balance, *INITIAL_BALANCE_RANGE, 4)
        * 0.1  # Encourage lower initial balances
        + exponential(
            final_balance_average_relative_median,
            min(1, final_balance_average_relative_min),
            min(1, final_balance_average_relative_max),
            5,
        )
        * 0.2  # Encourage ending with more than initial balance
    )

    trial.set_user_attr("prob_success", float(prob_success))
    trial.set_user_attr(
        "avg_relative_gain", float(final_balance_average_relative_median)
    )
    trial.set_user_attr("score", float(score))
    trial.set_user_attr(
        "final_balance_average_relative_median",
        float(final_balance_average_relative_median),
    )

    return score  # maximize this composite score


if __name__ == "__main__":
    # Try to load an existing study; create it if it doesn't exist
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
        print(f"Resuming existing study: {STUDY_NAME}")
    except KeyError:
        study = optuna.create_study(
            study_name=STUDY_NAME, storage=STORAGE_PATH, direction="maximize"
        )
        print(f"Created new study: {STUDY_NAME}")

    study.optimize(objective, n_trials=TRIAL_COUNT, n_jobs=1)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_n_sims = best_trial.user_attrs["n_sims_used"]

    print(f"\n=== Optimization Results for {STUDY_NAME} ===")
    print(f"Best Trial #{best_trial.number}")
    print(f"  Initial Balance: ${best_params['initial_balance']:,}")
    print(f"  Withdrawal: ${best_params['withdrawal']:,}")
    print(f"  Withdrawal Negative Year: ${best_params['withdrawal_negative_year']:,}")
    print(
        f"  Withdrawal percentage: {best_params['withdrawal_percentage_when_negative_year']:,}%"
    )
    print(
        f"  Probability of Success: {best_trial.user_attrs['prob_success'] * 100:.2f}%"
    )
    print(f"  Simulations Count used: {best_n_sims:,}")
    print(f"  Final Score: {best_trial.user_attrs['score']:.2%}")
    print(f"  Retirement years: {RETIREMENT_YEARS} years")
    print(
        f"  Random with real life constraints: {'Yes' if REAL_LIFE_CONSTRAINTS else 'No'}"
    )
    print(f"  SP500: {best_params['sp500_percentage']:.2%}, Bond rate: {BOND_RATE:.2%}")
    print(
        f"  Median relative final balance: {best_trial.user_attrs['final_balance_average_relative_median']:.2%} of initial balance"
    )
    # Leaderboard
    # TOP_N = 10
    # print("\n=== Top Leaderboard ===")
    # sorted_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    # for rank, t in enumerate(sorted_trials[:TOP_N], start=1):
    #     print(
    #         f"{rank:2d}. Score={t.value:.4f} | "
    #         f"Prob={t.user_attrs.get('prob_success', 0):,.3%} | "
    #         f"Init=${t.params['initial_balance']:,.0f} | "
    #         f"Wdrwl=${t.params['withdrawal']:,.0f} | "
    #         f"Wdrwl neg=${t.params['withdrawal_negative_year']:,.0f} | "
    #         f"SP500={t.params['sp500_percentage']:,.1%} | "
    #         f"Sims={t.user_attrs.get('n_sims_used', 0):,}"
    #     )

    # Optional: run final simulation on the best params
    print("\nRunning final validation simulation...")
    final_data = run_simulation_mp(
        n_sims=best_n_sims,
        initial_balance=best_params["initial_balance"],
        withdrawal=best_params["withdrawal"],
        withdrawal_negative_year=best_params["withdrawal_negative_year"],
        random_with_real_life_constraints=REAL_LIFE_CONSTRAINTS,
        sp500_percentage=best_params["sp500_percentage"],
        bond_rate=BOND_RATE,
        n_years=RETIREMENT_YEARS,
    )
    print(f"Final Probability of Success: {final_data.probability_of_success:.3%}")
    print(f"Standard Deviation: ${final_data.std_final:,.0f}")
    print(f"Standard error: ${final_data.std_error:,.0f}")
    print(
        f"Standard error / mean: {final_data.std_error / final_data.final_balances.mean():.3%}"
    )

    final_data.print_stats()
