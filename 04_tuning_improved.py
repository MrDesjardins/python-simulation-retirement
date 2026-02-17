import time
import optuna
import numpy as np
from common import exponential, inverse_exponential, run_simulation_mp

# Thresholds for adaptive simulation
STD_ERROR_DIFF_THRESHOLD = 0.01  # stop increasing n_sims if std_error stabilizes
STD_ERROR_ACCEPTANCE = 0.005  # accept std_error if small enough

# Constants
INITIAL_BALANCE_RANGE = (4_000_000, 7_000_000)
INITIAL_BALANCE_STEP = 200_000
WITHDRAWAL_RANGE = (80_000, 140_000)
WITHDRAWAL_STEP = 5_000
WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE = (0.80, 1.0)
WITHDRAWAL_NEGATIVE_STEP = 0.02
N_SIMS_RANGE = (50_000, 1_000_000)
STEP_N_SIMS = 50_000
TRIAL_COUNT = 5000
STORAGE_PATH = "sqlite:///db.sqlite3"
STUDY_NAME = (
    "retirement_tuning_study_v115"  # ⚠️ CHANGED: Fixed bugs + progressive batching + pruning
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
SUPPLEMENTAL_INCOME = 18_000  # per year

# Random seed configuration
# Set to None for production (explore different market scenarios each trial)
# Set to integer for debugging (reproducible results)
OPTIMIZATION_RANDOM_SEED = None  # None = explore different futures (RECOMMENDED)

# NEW: Configurable objective function weights
OBJECTIVE_WEIGHTS = {
    "prob_success": 0.50,      # Probability of portfolio survival
    "withdrawal": 0.15,        # Higher withdrawals preferred
    "initial_balance": 0.10,   # Lower initial balance preferred
    "withdrawal_consistency": 0.10,  # Consistent withdrawals in good/bad years
    "final_balance": 0.15,     # Better ending balance preferred
}

# NEW: Probability threshold range (widened from 0.99-1.0 to 0.90-1.0)
# This allows solutions with 95-99% success to be considered if they have other benefits
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
# Trials below this after initial batch are pruned (stopped early)
MIN_ACCEPTABLE_PROB = 0.85


def objective(trial):
    """
    Objective function for Optuna optimization.

    Improvements:
    - Removed redundant parameter (withdrawal_negative_year)
    - Progressive batching (don't throw away previous simulations)
    - Division by zero protection
    - Pruning support for early trial termination
    - Configurable weights and thresholds
    - Optional random seed (None = explore different futures, recommended)

    Note on random_seed:
    - If None (default): Each trial samples different market sequences
      This tests parameter robustness across different possible futures
    - If set (e.g., 42): All trials see same sequences (only for debugging)
      This reduces exploration but makes results reproducible
    - Recommended: Use None for production optimization
    """
    # Suggest parameters (one less than before - removed redundant withdrawal_negative_year)
    initial_balance: float = trial.suggest_float(
        "initial_balance", *INITIAL_BALANCE_RANGE, step=INITIAL_BALANCE_STEP
    )
    withdrawal: float = trial.suggest_float(
        "withdrawal", *WITHDRAWAL_RANGE, step=WITHDRAWAL_STEP
    )

    # FIXED: Only suggest percentage, compute actual withdrawal from it
    withdrawal_percentage_when_negative_year: float = trial.suggest_float(
        "withdrawal_percentage_when_negative_year",
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[0],
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[1],
        step=WITHDRAWAL_NEGATIVE_STEP,
    )
    # Compute withdrawal_negative_year (no longer a separate parameter)
    withdrawal_negative_year: float = withdrawal * withdrawal_percentage_when_negative_year

    sp500_percentage: float = trial.suggest_float(
        "sp500_percentage",
        PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN,
        PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX,
        step=PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP,
    )

    # NEW: Progressive batching - accumulate results instead of re-running
    n_sims_done = 0
    all_final_balances = []
    last_std_error = float("inf")

    while n_sims_done < N_SIMS_RANGE[1]:
        # Run next batch
        batch_size = min(STEP_N_SIMS, N_SIMS_RANGE[1] - n_sims_done)

        simulation_data = run_simulation_mp(
            n_sims=batch_size,
            initial_balance=initial_balance,
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
            random_seed=OPTIMIZATION_RANDOM_SEED,  # None = different futures per trial
        )

        # Accumulate results
        all_final_balances.append(simulation_data.final_balances)
        n_sims_done += batch_size

        # Compute cumulative statistics
        cumulative_balances = np.concatenate(all_final_balances)
        prob_success = float(np.mean(cumulative_balances > 0))

        # NEW: Pruning - stop early if probability too low
        if n_sims_done >= N_SIMS_RANGE[0]:  # Only prune after minimum simulations
            trial.report(prob_success, step=n_sims_done)
            if trial.should_prune():
                trial.set_user_attr("n_sims_used", n_sims_done)
                raise optuna.TrialPruned()

            # Also prune if below absolute minimum
            if prob_success < MIN_ACCEPTABLE_PROB:
                trial.set_user_attr("n_sims_used", n_sims_done)
                raise optuna.TrialPruned()

        # Compute convergence metrics
        std_final = np.std(cumulative_balances, ddof=1)
        std_error = std_final / np.sqrt(n_sims_done)

        # FIXED: Division by zero protection
        mean_balance = np.mean(cumulative_balances)
        if mean_balance > 0:
            std_error_relative_to_mean = std_error / mean_balance
        else:
            std_error_relative_to_mean = float("inf")  # Force more simulations

        diff_compare_last = (
            (last_std_error - std_error) / last_std_error
            if last_std_error != float("inf")
            else float("inf")
        )

        # Check convergence
        if std_error_relative_to_mean <= STD_ERROR_ACCEPTANCE or (
            diff_compare_last != float("inf")
            and diff_compare_last <= STD_ERROR_DIFF_THRESHOLD
        ):
            break

        last_std_error = std_error

    # Store how many simulations were used for this trial
    trial.set_user_attr("n_sims_used", n_sims_done)

    # Use final cumulative results
    final_balances = np.concatenate(all_final_balances)
    prob_success = float(np.mean(final_balances > 0))
    final_balance_growth_ratios = final_balances / initial_balance
    final_balance_growth_ratios_median = float(np.median(final_balance_growth_ratios))

    # IMPROVED: Wider probability threshold (0.90-1.0 instead of 0.99-1.0)
    prob_term = exponential(
        prob_success,
        PROB_THRESHOLD_MIN,
        PROB_THRESHOLD_MAX,
        PROB_THRESHOLD_STEEPNESS
    )

    # Encourage higher withdrawals
    withdrawal_term = exponential(withdrawal, *WITHDRAWAL_RANGE, 8)

    # Encourage lower starting balances
    initial_balance_term = inverse_exponential(
        initial_balance, *INITIAL_BALANCE_RANGE, 2
    )

    # Encourage small difference between negative year and normal withdrawal
    withdrawal_diff_ratio = (withdrawal - withdrawal_negative_year) / withdrawal
    withdrawal_diff_ratio_term = inverse_exponential(withdrawal_diff_ratio, 0.0, 1.0, 3)

    # Strongly reward ending with more than initial
    final_balance_term = exponential(
        np.clip(final_balance_growth_ratios_median, 0.0, 10.0),
        0,
        10.0,
        10,
    )

    # NEW: Use configurable weights
    score = (
        prob_term * OBJECTIVE_WEIGHTS["prob_success"]
        + withdrawal_term * OBJECTIVE_WEIGHTS["withdrawal"]
        + initial_balance_term * OBJECTIVE_WEIGHTS["initial_balance"]
        + withdrawal_diff_ratio_term * OBJECTIVE_WEIGHTS["withdrawal_consistency"]
        + final_balance_term * OBJECTIVE_WEIGHTS["final_balance"]
    )

    # Save diagnostics
    trial.set_user_attr("prob_success", prob_success)
    trial.set_user_attr("prob_term", float(prob_term))
    trial.set_user_attr("withdrawal_term", float(withdrawal_term))
    trial.set_user_attr("withdrawal_negative_year", float(withdrawal_negative_year))
    trial.set_user_attr("initial_balance_term", float(initial_balance_term))
    trial.set_user_attr("withdrawal_diff_ratio_term", float(withdrawal_diff_ratio_term))
    trial.set_user_attr("final_balance_term", float(final_balance_term))
    trial.set_user_attr(
        "final_relative_balance_to_median", float(final_balance_growth_ratios_median)
    )
    trial.set_user_attr("score", float(score))

    return score


if __name__ == "__main__":
    start_time = time.perf_counter()

    # Try to load an existing study; create it if it doesn't exist
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
        print(f"Resuming existing study: {STUDY_NAME}")
    except KeyError:
        # NEW: Create study with MedianPruner for early stopping
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=STORAGE_PATH,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,  # Don't prune first 5 trials
                n_warmup_steps=1,     # Start pruning after 1 batch (50k sims)
            ),
        )
        print(f"Created new study: {STUDY_NAME}")

    # Optimize (still n_jobs=1 since each trial uses multiprocessing internally)
    study.optimize(objective, n_trials=TRIAL_COUNT, n_jobs=1)

    best_trial = study.best_trial
    best_params = best_trial.params

    # Calculate total simulations including pruned trials
    sum_n_sims_all_trials = sum(
        t.user_attrs.get("n_sims_used", 0) for t in study.trials
    )
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    # Calculate efficiency metrics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    avg_sims_per_trial = sum_n_sims_all_trials / len(study.trials) if len(study.trials) > 0 else 0

    # Calculate explored parameter space
    n_initial_balance = round((INITIAL_BALANCE_RANGE[1] - INITIAL_BALANCE_RANGE[0]) / INITIAL_BALANCE_STEP) + 1
    n_withdrawal = round((WITHDRAWAL_RANGE[1] - WITHDRAWAL_RANGE[0]) / WITHDRAWAL_STEP) + 1
    n_withdrawal_pct = round((WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[1] - WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[0]) / WITHDRAWAL_NEGATIVE_STEP) + 1
    n_sp500 = round((PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX - PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN) / PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP) + 1
    total_combinations = n_initial_balance * n_withdrawal * n_withdrawal_pct * n_sp500
    explored_pct = len(study.trials) / total_combinations * 100

    print(f"\n=== Optimization Summary ===")
    print(f"Total trials run: {len(study.trials)}")
    print(f"  - Completed: {len(completed_trials)}")
    print(f"  - Pruned (stopped early): {pruned_count}")
    print(f"Total simulations across all trials: {sum_n_sims_all_trials:,}")
    print(f"Average simulations per trial: {avg_sims_per_trial:,.0f}")
    print(f"Parameter space explored: {explored_pct:.1f}% ({len(study.trials):,} trials / {total_combinations:,} combinations)")
    print(f"  - initial_balance: {n_initial_balance} values, withdrawal: {n_withdrawal} values, withdrawal_pct: {n_withdrawal_pct} values, sp500: {n_sp500} values")

    print(f"\n=== Optimization Results for {STUDY_NAME} ===")
    print(f"Best Trial #{best_trial.number}")
    print(f"  Initial Balance: ${best_params['initial_balance']:,}")
    print(f"  Withdrawal: ${best_params['withdrawal']:,}")

    # Compute withdrawal_negative_year from percentage
    withdrawal_negative_year = (
        best_params['withdrawal'] *
        best_params['withdrawal_percentage_when_negative_year']
    )
    print(f"  Withdrawal Negative Year: ${withdrawal_negative_year:,}")
    print(
        f"  Withdrawal ratio: {best_params['withdrawal_percentage_when_negative_year']:.2f}"
    )
    print(f"  Retirement years: {RETIREMENT_YEARS} years")
    print(
        f"  Random with real life constraints: {'Yes' if REAL_LIFE_CONSTRAINTS else 'No'}"
    )
    print(
        f"  SP500: {best_params['sp500_percentage']:.2%}, Bond rate: {BOND_RATE:.2%}, Inflation rate: {INFLATION_RATE:.2%}"
    )
    print(f"  Social Security starts after {YEARS_WITHOUT_SOCIAL_SECURITY} years, amount: ${SOCIAL_SECURITY_MONEY:,} per year")
    print(f"  Supplemental Income for {YEARS_WITH_SUPPLEMENTAL_INCOME} years, amount: ${SUPPLEMENTAL_INCOME:,} per year")
    print("\n~~~~~Score Details~~~~~")
    print("Score Components:")
    print(
        f"  Probability of Success: {best_trial.user_attrs['prob_success'] * 100:.2f}%"
    )
    print(f"  Final Score: {best_trial.user_attrs['score']:.2%}")
    print(f"  Simulations used for best trial: {best_trial.user_attrs['n_sims_used']:,}")
    print(f"  Total simulations (all {len(study.trials)} trials): {sum_n_sims_all_trials:,}")
    print(
        f"  Median relative final balance: {best_trial.user_attrs['final_relative_balance_to_median']:.2%} of initial balance"
    )
    print("\nObjective Terms:")
    print(f"  Prob Success: {best_trial.user_attrs['prob_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['prob_success']:.2f})")
    print(f"  Withdraw: {best_trial.user_attrs['withdrawal_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['withdrawal']:.2f})")
    print(f"  Init Balance: {best_trial.user_attrs['initial_balance_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['initial_balance']:.2f})")
    print(f"  Withdraw Consistency: {best_trial.user_attrs['withdrawal_diff_ratio_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['withdrawal_consistency']:.2f})")
    print(f"  Final Balance: {best_trial.user_attrs['final_balance_term']:.4f} (weight: {OBJECTIVE_WEIGHTS['final_balance']:.2f})")

    # Final validation uses a fixed high count for a precise, definitive result
    # (independent of how many sims the best trial happened to use during optimization)
    FINAL_VALIDATION_N_SIMS = 1_000_000
    print(f"\n~~~~~Running final validation simulation ({FINAL_VALIDATION_N_SIMS:,} sims)~~~~~")
    final_data = run_simulation_mp(
        n_sims=FINAL_VALIDATION_N_SIMS,
        initial_balance=best_params["initial_balance"],
        withdrawal=best_params["withdrawal"],
        withdrawal_negative_year=withdrawal_negative_year,
        random_with_real_life_constraints=REAL_LIFE_CONSTRAINTS,
        sp500_percentage=best_params["sp500_percentage"],
        bond_rate=BOND_RATE,
        n_years=RETIREMENT_YEARS,
        inflation_rate=INFLATION_RATE,
        years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
        social_security_money=SOCIAL_SECURITY_MONEY,
        years_with_supplemental_income=YEARS_WITH_SUPPLEMENTAL_INCOME,  # FIXED: Was missing
        supplemental_income=SUPPLEMENTAL_INCOME,  # FIXED: Was missing
        random_seed=OPTIMIZATION_RANDOM_SEED,  # None = realistic validation
    )
    print(f"Final Probability of Success: {final_data.probability_of_success:.3%}")
    print(f"Standard Deviation: ${final_data.std_final:,.0f}")
    print(f"Standard error: ${final_data.std_error:,.0f}")

    # FIXED: Division by zero protection
    mean_final = final_data.final_balances.mean()
    if mean_final > 0:
        print(f"Relative Standard Error: {final_data.std_error / mean_final:.3%}")
    else:
        print(f"Relative Standard Error: N/A (mean balance is zero)")

    final_data.print_stats()
    print("\n=== End of optimization ===")
    end_time = time.perf_counter()
    print(f"Total optimization time: {end_time - start_time:.2f} seconds")
