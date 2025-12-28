import time
import optuna
import numpy as np
from common import exponential, inverse_exponential, run_simulation_mp

# Thresholds for adaptive simulation
STD_ERROR_DIFF_THRESHOLD = 0.01  # stop increasing n_sims if std_error stabilizes
STD_ERROR_ACCEPTANCE = 0.005  # accept std_error if small enough

# Constants
INITIAL_BALANCE_RANGE = (4_000_000, 10_000_000)
INITIAL_BALANCE_STEP = 200_000
WITHDRAWAL_RANGE = (75_000, 130_000)
WITHDRAWAL_STEP = 5_000
WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE = (0.80, 1.0)
WITHDRAWAL_NEGATIVE_STEP = 0.02
N_SIMS_RANGE = (50_000, 500_000)
STEP_N_SIMS = 50_000
TRIAL_COUNT = 250
STORAGE_PATH = "sqlite:///db.sqlite3"
STUDY_NAME = (
    "retirement_tuning_study_v107"  # ⚠️ CHANGE EVERYTIME WE CHANGE CONSTANTS OR LOGIC ⚠️
)
REAL_LIFE_CONSTRAINTS = False
RETIREMENT_YEARS = 40
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN = 0.7
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX = 1.0
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP = 0.05
INFLATION_RATE = 0.03 # Vanguard projection 10 years worse case as of November 2025: 0.026
BOND_RATE = 0.03  # Bond rate for 2 years as of November 2025: 0.034

YEARS_WITHOUT_SOCIAL_SECURITY = 20
SOCIAL_SECURITY_MONEY = 50_000  # per year

YEARS_WITH_SUPPLEMENTAL_INCOME = 12 # Spouse working
SUPPLEMENTAL_INCOME = 30_000  # per year

def objective(trial):
    # Suggest both initial balance and withdrawal amount
    initial_balance: float = trial.suggest_float(
        "initial_balance", *INITIAL_BALANCE_RANGE, step=INITIAL_BALANCE_STEP
    )
    withdrawal: float = trial.suggest_float(
        "withdrawal", *WITHDRAWAL_RANGE, step=WITHDRAWAL_STEP
    )
    withdrawal_percentage_when_negative_year: float = trial.suggest_float(
        "withdrawal_percentage_when_negative_year",
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[0],
        WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE[1],
        step=WITHDRAWAL_NEGATIVE_STEP,
    )
    withdrawal_negative_year: float = trial.suggest_float(
        "withdrawal_negative_year",
        withdrawal * withdrawal_percentage_when_negative_year,
        withdrawal,
        step=WITHDRAWAL_STEP,
    )
    sp500_percentage: float = trial.suggest_float(
        "sp500_percentage",
        PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN,
        PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX,
        step=PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP,
    )

    # Adaptive Monte Carlo configuration (unchanged)
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
            years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
            social_security_money=SOCIAL_SECURITY_MONEY,
            years_with_supplemental_income=YEARS_WITH_SUPPLEMENTAL_INCOME,
            supplemental_income=SUPPLEMENTAL_INCOME,
        )

        std_error = simulation_data.std_error
        diff_compare_last = (
            (last_std_error - std_error) / last_std_error
            if last_std_error != float("inf")
            else float("inf")
        )

        std_error_relative_to_mean = std_error / simulation_data.final_balances.mean()

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
    prob_success = float(simulation_data.probability_of_success)

    final_balances = simulation_data.final_balances
    if final_balances.ndim == 2:
        last_year_balances = final_balances[:, -1]
    else:
        last_year_balances = final_balances
    final_balance_growth_ratios = last_year_balances / initial_balance

    # Robust statistics for scaling (avoid per-trial min/max and outliers):
    # use percentile bounds so mapping is stable across similar trials
    vmin = float(np.percentile(final_balance_growth_ratios, 5))
    vmax = float(np.percentile(final_balance_growth_ratios, 95))

    # If percentiles collapse (rare), fallback to sensible defaults relative to initial_balance
    # e.g. allow values from 0 (complete loss) up to 3x initial
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
        vmin = 0.0
        vmax = max(1.0, np.median(final_balance_growth_ratios) * 2.0)

    final_balance_growth_ratios_median = float(np.median(final_balance_growth_ratios))

    # Probability term (differentiates high 0.9–1 range)
    prob_term = exponential(prob_success, 0.99, 1.0, 7)

    # Encourage higher withdrawals slightly
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

    score = (
        prob_term * 0.50
        + withdrawal_term * 0.15
        + initial_balance_term * 0.10
        + withdrawal_diff_ratio_term * 0.10
        + final_balance_term * 0.15
    )

    # Save diagnostics so we can inspect importance of each term
    trial.set_user_attr("prob_success", prob_success)
    trial.set_user_attr("prob_term", float(prob_term))
    trial.set_user_attr("withdrawal_term", float(withdrawal_term))
    trial.set_user_attr("withdrawal_negative_year", float(withdrawal_negative_year))
    trial.set_user_attr("initial_balance_term", float(initial_balance_term))
    trial.set_user_attr("withdrawal_diff_ratio_term", float(withdrawal_diff_ratio_term))
    trial.set_user_attr("final_balance_term", float(final_balance_term))
    trial.set_user_attr("vmin_final_balance", float(vmin))
    trial.set_user_attr("vmax_final_balance", float(vmax))
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
        study = optuna.create_study(
            study_name=STUDY_NAME, storage=STORAGE_PATH, direction="maximize"
        )
        print(f"Created new study: {STUDY_NAME}")

    study.optimize(objective, n_trials=TRIAL_COUNT, n_jobs=1)

    best_trial = study.best_trial
    best_params = best_trial.params
    sum_n_sims_all_trials = sum(
        t.user_attrs.get("n_sims_used", 0) for t in study.trials
    )
    print(f"\nTotal simulations run across all trials: {sum_n_sims_all_trials:,}")
    print(f"\n=== Optimization Results for {STUDY_NAME} ===")
    print(f"Best Trial #{best_trial.number}")
    print(f"  Initial Balance: ${best_params['initial_balance']:,}")
    print(f"  Withdrawal: ${best_params['withdrawal']:,}")
    print(f"  Withdrawal Negative Year: ${best_params['withdrawal_negative_year']:,}")
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
    print(f"  Simulations Count used: {best_trial.user_attrs['n_sims_used']:,}")
    print(
        f"  Median relative final balance: {best_trial.user_attrs['final_relative_balance_to_median']:.2%} of initial balance"
    )
    print("Terms:")
    print(f"  Prob Success: {best_trial.user_attrs['prob_term']:.4f}")
    print(f"  Withdraw: {best_trial.user_attrs['withdrawal_term']:.4f}")
    print(f"  Init Balance: {best_trial.user_attrs['initial_balance_term']:.4f}")
    print(
        f"  Withdraw Diff Ratio: {best_trial.user_attrs['withdrawal_diff_ratio_term']:.4f}"
    )
    print(f"  Final Balance: {best_trial.user_attrs['final_balance_term']:.4f}")

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
    print("\n~~~~~Running final validation simulation~~~~~")
    final_data = run_simulation_mp(
        n_sims=best_trial.user_attrs["n_sims_used"],
        initial_balance=best_params["initial_balance"],
        withdrawal=best_params["withdrawal"],
        withdrawal_negative_year=best_params["withdrawal_negative_year"],
        random_with_real_life_constraints=REAL_LIFE_CONSTRAINTS,
        sp500_percentage=best_params["sp500_percentage"],
        bond_rate=BOND_RATE,
        n_years=RETIREMENT_YEARS,
        inflation_rate=INFLATION_RATE,
        years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
        social_security_money=SOCIAL_SECURITY_MONEY,
    )
    print(f"Final Probability of Success: {final_data.probability_of_success:.3%}")
    print(f"Standard Deviation: ${final_data.std_final:,.0f}")
    print(f"Standard error: ${final_data.std_error:,.0f}")
    print(
        f"Relative Standard Error: {final_data.std_error / final_data.final_balances.mean():.3%}"
    )

    final_data.print_stats()
    print("\n=== End of optimization ===")
    end_time = time.perf_counter()
    print(f"Total optimization time: {end_time - start_time:.2f} seconds")
