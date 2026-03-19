import time
import optuna
import numpy as np
from common import exponential, inverse_exponential, run_simulation_mp

# Thresholds for adaptive simulation
STD_ERROR_DIFF_THRESHOLD = 0.01  # stop increasing n_sims if std_error stabilizes
STD_ERROR_ACCEPTANCE = 0.005  # accept std_error if small enough

# Constants
INITIAL_BALANCE = 3_600_000  # Fixed — this is the known starting capital
WITHDRAWAL_RANGE = (60_000, 120_000)
WITHDRAWAL_STEP = 2_500
WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE = (0.90, 1.0)
WITHDRAWAL_NEGATIVE_STEP = 0.01
N_SIMS_RANGE = (50_000, 1_000_000)
STEP_N_SIMS = 50_000
TRIAL_COUNT = 3_000
STORAGE_PATH = "sqlite:///db_05.sqlite3"  # Separate DB — does not affect study 04
STUDY_NAME = (
    "target_budget_study_v3"  # Independent study from retirement_tuning_study_*
)
REAL_LIFE_CONSTRAINTS = False
RETIREMENT_YEARS = 40
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN = 0.7
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX = 1.0
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP = 0.05
INFLATION_RATE = 0.03  # Vanguard projection 10 years worse case as of November 2025: 0.026
BOND_RATE = 0.03  # Bond rate for 2 years as of November 2025: 0.034

YEARS_WITHOUT_SOCIAL_SECURITY = 20
SOCIAL_SECURITY_MONEY = 40_000  # per year

YEARS_WITH_SUPPLEMENTAL_INCOME = 15  # Spouse working
SUPPLEMENTAL_INCOME = 20_000  # per year

# Random seed configuration
# Set to None for production (explore different market scenarios each trial)
# Set to integer for debugging (reproducible results)
OPTIMIZATION_RANDOM_SEED = None  # None = explore different futures (RECOMMENDED)

# Objective function weights
# initial_balance removed — it is fixed, not optimized
# Emphasis shifted toward withdrawal (main goal) and probability of success (safety)
OBJECTIVE_WEIGHTS = {
    "prob_success": 0.55,          # Probability of portfolio survival — primary safety constraint
    "withdrawal": 0.25,            # Higher withdrawals preferred — this is the main goal
    "withdrawal_consistency": 0.10, # Consistent withdrawals in good/bad years
    "final_balance": 0.10,         # Better ending balance preferred
}

# Probability threshold range
PROB_THRESHOLD_MIN = 0.95
PROB_THRESHOLD_MAX = 1.0
#   ┌───────────┬───────────┬───────┬───────┐
#   │ Success % │ k=1       │  k=5  │ k=10  │
#   ├───────────┼───────────┼───────┼───────┤
#   │ 95%       │ ~0.00     │ ~0.00 │ ~0.00 │
#   ├───────────┼───────────┼───────┼───────┤
#   │ 97%       │ ~0.37     │ ~0.73 │ ~0.93 │
#   ├───────────┼───────────┼───────┼───────┤
#   │ 99%       │ ~0.71     │ ~0.97 │ ~1.00 │
#   ├───────────┼───────────┼───────┼───────┤
#   │ 100%      │ 1.00      │ 1.00  │ 1.00  │
#   └───────────┴───────────┴───────┴───────┘
PROB_THRESHOLD_STEEPNESS = 6

# Pruning: Minimum acceptable probability to continue trial
MIN_ACCEPTABLE_PROB = 0.85


def objective(trial):
    """
    Objective function for finding the highest safe annual withdrawal
    for a fixed initial balance.

    Unlike 04_tuning_improved.py, initial_balance is a known constant here.
    The optimizer seeks to maximize withdrawal while remaining safe (high
    probability of success) and maintaining other quality criteria.

    Note on random_seed:
    - If None (default): Each trial samples different market sequences.
      This tests parameter robustness across different possible futures.
    - If set (e.g., 42): All trials see same sequences (only for debugging).
    - Recommended: Use None for production optimization.
    """
    withdrawal: float = trial.suggest_float(
        "withdrawal", *WITHDRAWAL_RANGE, step=WITHDRAWAL_STEP
    )

    withdrawal_percentage_when_negative_year: float = trial.suggest_float(
        "withdrawal_percentage_when_negative_year",
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[0],
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[1],
        step=WITHDRAWAL_NEGATIVE_STEP,
    )
    withdrawal_negative_year: float = withdrawal * withdrawal_percentage_when_negative_year

    sp500_percentage: float = trial.suggest_float(
        "sp500_percentage",
        PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN,
        PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX,
        step=PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP,
    )

    # Progressive batching — accumulate results instead of re-running
    n_sims_done = 0
    all_final_balances = []
    last_std_error = float("inf")

    while n_sims_done < N_SIMS_RANGE[1]:
        batch_size = min(STEP_N_SIMS, N_SIMS_RANGE[1] - n_sims_done)

        simulation_data = run_simulation_mp(
            n_sims=batch_size,
            initial_balance=INITIAL_BALANCE,
            random_with_real_life_constraints=REAL_LIFE_CONSTRAINTS,
            withdrawal=withdrawal,
            withdrawal_negative_year=withdrawal_negative_year,
            n_years=RETIREMENT_YEARS,
            sp500_percentage=sp500_percentage,
            bond_rate=BOND_RATE,
            inflation_rate=INFLATION_RATE,
            years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
            social_security_money=SOCIAL_SECURITY_MONEY,
            years_with_supplemental_income=YEARS_WITH_SUPPLEMENTAL_INCOME,
            supplemental_income=SUPPLEMENTAL_INCOME,
            random_seed=OPTIMIZATION_RANDOM_SEED,
        )

        all_final_balances.append(simulation_data.final_balances)
        n_sims_done += batch_size

        cumulative_balances = np.concatenate(all_final_balances)
        prob_success = float(np.mean(cumulative_balances > 0))

        # Pruning — stop early if probability too low
        if n_sims_done >= N_SIMS_RANGE[0]:
            trial.report(prob_success, step=n_sims_done)
            if trial.should_prune():
                trial.set_user_attr("n_sims_used", n_sims_done)
                raise optuna.TrialPruned()

            if prob_success < MIN_ACCEPTABLE_PROB:
                trial.set_user_attr("n_sims_used", n_sims_done)
                raise optuna.TrialPruned()

        # Convergence check
        std_final = np.std(cumulative_balances, ddof=1)
        std_error = std_final / np.sqrt(n_sims_done)

        mean_balance = np.mean(cumulative_balances)
        if mean_balance > 0:
            std_error_relative_to_mean = std_error / mean_balance
        else:
            std_error_relative_to_mean = float("inf")

        diff_compare_last = (
            (last_std_error - std_error) / last_std_error
            if last_std_error != float("inf")
            else float("inf")
        )

        if std_error_relative_to_mean <= STD_ERROR_ACCEPTANCE or (
            diff_compare_last != float("inf")
            and diff_compare_last <= STD_ERROR_DIFF_THRESHOLD
        ):
            break

        last_std_error = std_error

    trial.set_user_attr("n_sims_used", n_sims_done)

    final_balances = np.concatenate(all_final_balances)
    prob_success = float(np.mean(final_balances > 0))
    final_balance_growth_ratios = final_balances / INITIAL_BALANCE
    final_balance_growth_ratios_median = float(np.median(final_balance_growth_ratios))

    prob_term = exponential(
        prob_success,
        PROB_THRESHOLD_MIN,
        PROB_THRESHOLD_MAX,
        PROB_THRESHOLD_STEEPNESS,
    )

    # Encourage higher withdrawals — primary optimization goal
    withdrawal_term = exponential(withdrawal, *WITHDRAWAL_RANGE, 8)

    # Encourage small difference between negative year and normal withdrawal
    withdrawal_diff_ratio = (withdrawal - withdrawal_negative_year) / withdrawal
    withdrawal_diff_ratio_term = inverse_exponential(withdrawal_diff_ratio, 0.0, 1.0, 3)

    # Reward ending with more than initial balance
    final_balance_term = exponential(
        np.clip(final_balance_growth_ratios_median, 0.0, 10.0),
        0,
        10.0,
        10,
    )

    score = (
        prob_term * OBJECTIVE_WEIGHTS["prob_success"]
        + withdrawal_term * OBJECTIVE_WEIGHTS["withdrawal"]
        + withdrawal_diff_ratio_term * OBJECTIVE_WEIGHTS["withdrawal_consistency"]
        + final_balance_term * OBJECTIVE_WEIGHTS["final_balance"]
    )

    # Save diagnostics
    trial.set_user_attr("prob_success", prob_success)
    trial.set_user_attr("prob_term", float(prob_term))
    trial.set_user_attr("withdrawal_term", float(withdrawal_term))
    trial.set_user_attr("withdrawal_negative_year", float(withdrawal_negative_year))
    trial.set_user_attr("withdrawal_diff_ratio_term", float(withdrawal_diff_ratio_term))
    trial.set_user_attr("final_balance_term", float(final_balance_term))
    trial.set_user_attr(
        "final_relative_balance_to_median", float(final_balance_growth_ratios_median)
    )
    trial.set_user_attr("score", float(score))

    return score


if __name__ == "__main__":
    start_time = time.perf_counter()

    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
        print(f"Resuming existing study: {STUDY_NAME}")
    except KeyError:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=STORAGE_PATH,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
            ),
        )
        print(f"Created new study: {STUDY_NAME}")

    study.optimize(objective, n_trials=TRIAL_COUNT, n_jobs=1)

    best_trial = study.best_trial
    best_params = best_trial.params

    # Summary statistics
    sum_n_sims_all_trials = sum(
        t.user_attrs.get("n_sims_used", 0) for t in study.trials
    )
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    avg_sims_per_trial = sum_n_sims_all_trials / len(study.trials) if len(study.trials) > 0 else 0

    n_withdrawal = round((WITHDRAWAL_RANGE[1] - WITHDRAWAL_RANGE[0]) / WITHDRAWAL_STEP) + 1
    n_withdrawal_pct = round((WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[1] - WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[0]) / WITHDRAWAL_NEGATIVE_STEP) + 1
    n_sp500 = round((PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX - PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN) / PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP) + 1
    total_combinations = n_withdrawal * n_withdrawal_pct * n_sp500
    explored_pct = len(study.trials) / total_combinations * 100

    print(f"\n=== Optimization Summary ===")
    print(f"Total trials run: {len(study.trials)}")
    print(f"  - Completed: {len(completed_trials)}")
    print(f"  - Pruned (stopped early): {pruned_count}")
    print(f"Total simulations across all trials: {sum_n_sims_all_trials:,}")
    print(f"Average simulations per trial: {avg_sims_per_trial:,.0f}")
    print(f"Parameter space explored: {explored_pct:.1f}% ({len(study.trials):,} trials / {total_combinations:,} combinations)")
    print(f"  - withdrawal: {n_withdrawal} values, withdrawal_pct: {n_withdrawal_pct} values, sp500: {n_sp500} values")

    withdrawal_negative_year = (
        best_params["withdrawal"]
        * best_params["withdrawal_percentage_when_negative_year"]
    )

    print(f"\n=== Optimization Results for {STUDY_NAME} ===")
    print(f"Best Trial #{best_trial.number}")
    print(f"  Initial Balance (fixed): ${INITIAL_BALANCE:,}")
    print(f"  Withdrawal: ${best_params['withdrawal']:,}")
    print(f"  Withdrawal Negative Year: ${withdrawal_negative_year:,}")
    print(f"  Withdrawal ratio: {best_params['withdrawal_percentage_when_negative_year']:.2f}")
    print(f"  Withdrawal as % of initial balance: {best_params['withdrawal'] / INITIAL_BALANCE:.2%}")
    print(f"  Retirement years: {RETIREMENT_YEARS} years")
    print(f"  Random with real life constraints: {'Yes' if REAL_LIFE_CONSTRAINTS else 'No'}")
    print(
        f"  SP500: {best_params['sp500_percentage']:.2%}, Bond rate: {BOND_RATE:.2%}, Inflation rate: {INFLATION_RATE:.2%}"
    )
    print(f"  Social Security starts after {YEARS_WITHOUT_SOCIAL_SECURITY} years, amount: ${SOCIAL_SECURITY_MONEY:,} per year")
    print(f"  Supplemental Income for {YEARS_WITH_SUPPLEMENTAL_INCOME} years, amount: ${SUPPLEMENTAL_INCOME:,} per year")
    print("\n~~~~~Score Details~~~~~")
    print("Score Components:")
    print(f"  Probability of Success: {best_trial.user_attrs['prob_success'] * 100:.2f}%")
    print(f"  Final Score: {best_trial.user_attrs['score']:.2%}")
    print(f"  Simulations used for best trial: {best_trial.user_attrs['n_sims_used']:,}")
    print(f"  Total simulations (all {len(study.trials)} trials): {sum_n_sims_all_trials:,}")
    print(f"  Median relative final balance: {best_trial.user_attrs['final_relative_balance_to_median']:.2%} of initial balance")
    print("\nObjective Terms:")
    print(f"  Prob Success: {best_trial.user_attrs['prob_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['prob_success']:.2f})")
    print(f"  Withdrawal: {best_trial.user_attrs['withdrawal_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['withdrawal']:.2f})")
    print(f"  Withdraw Consistency: {best_trial.user_attrs['withdrawal_diff_ratio_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['withdrawal_consistency']:.2f})")
    print(f"  Final Balance: {best_trial.user_attrs['final_balance_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['final_balance']:.2f})")

    FINAL_VALIDATION_N_SIMS = 1_000_000
    print(f"\n~~~~~Running final validation simulation ({FINAL_VALIDATION_N_SIMS:,} sims)~~~~~")
    final_data = run_simulation_mp(
        n_sims=FINAL_VALIDATION_N_SIMS,
        initial_balance=INITIAL_BALANCE,
        withdrawal=best_params["withdrawal"],
        withdrawal_negative_year=withdrawal_negative_year,
        random_with_real_life_constraints=REAL_LIFE_CONSTRAINTS,
        sp500_percentage=best_params["sp500_percentage"],
        bond_rate=BOND_RATE,
        n_years=RETIREMENT_YEARS,
        inflation_rate=INFLATION_RATE,
        years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
        social_security_money=SOCIAL_SECURITY_MONEY,
        years_with_supplemental_income=YEARS_WITH_SUPPLEMENTAL_INCOME,
        supplemental_income=SUPPLEMENTAL_INCOME,
        random_seed=OPTIMIZATION_RANDOM_SEED,
    )
    print(f"Final Probability of Success: {final_data.probability_of_success:.3%}")
    print(f"Standard Deviation: ${final_data.std_final:,.0f}")
    print(f"Standard error: ${final_data.std_error:,.0f}")

    mean_final = final_data.final_balances.mean()
    if mean_final > 0:
        print(f"Relative Standard Error: {final_data.std_error / mean_final:.3%}")
    else:
        print(f"Relative Standard Error: N/A (mean balance is zero)")

    final_data.print_stats()
    print("\n=== End of optimization ===")
    end_time = time.perf_counter()
    print(f"Total optimization time: {end_time - start_time:.2f} seconds")
