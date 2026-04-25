import time
import optuna
import numpy as np
from common import exponential, format_withdrawal_breakdown, inverse_exponential, run_simulation_mp
from simulation_convergence import (
    MIN_SIMS_FOR_CONVERGENCE,
    PROB_SUCCESS_SE_ACCEPTANCE,
    PROB_SUCCESS_SE_DIFF_THRESHOLD,
    cumulative_success_metrics,
    prob_convergence_should_stop,
)
from simulation_scenario import (
    TUNING_BLOCK_BOOTSTRAP_SIZE,
    TUNING_BOND_RATE,
    TUNING_INFLATION_RATE,
    TUNING_RETIREMENT_YEARS,
    TUNING_SAMPLING_MODE,
    tuning_run_simulation_common_kwargs,
)
from tuning_run_results import (
    build_run_results,
    export_run_results_sidecars,
    safe_best_trial,
    simulation_data_summary,
    trials_to_dataframe,
)

# Thresholds for adaptive simulation (success-rate SE, not mean-balance SE).
# See simulation_convergence.py.

# Constants
INITIAL_BALANCE_RANGE = (4_000_000, 10_000_000)
INITIAL_BALANCE_STEP = 200_000
WITHDRAWAL_RANGE = (75_000, 130_000)
WITHDRAWAL_STEP = 5_000
WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE = (0.80, 1.0)
WITHDRAWAL_NEGATIVE_STEP = 0.02
N_SIMS_RANGE = (50_000, 1_500_000)  # higher ceiling for convergence
STEP_N_SIMS = 50_000
TRIAL_COUNT = 250
STORAGE_PATH = "sqlite:///db.sqlite3"
STUDY_NAME = (
    "retirement_tuning_study_v108"  # ⚠️ v108: nominal returns, block bootstrap, tighter convergence
)
RESULTS_JSON_PATH = f"results/{STUDY_NAME}_meta.json"
SAMPLING_MODE = TUNING_SAMPLING_MODE
BLOCK_BOOTSTRAP_SIZE = TUNING_BLOCK_BOOTSTRAP_SIZE
RETIREMENT_YEARS = TUNING_RETIREMENT_YEARS
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN = 0.7
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX = 1.0
PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP = 0.05
INFLATION_RATE = TUNING_INFLATION_RATE
BOND_RATE = TUNING_BOND_RATE

YEARS_WITHOUT_SOCIAL_SECURITY = 20
SOCIAL_SECURITY_MONEY = 50_000  # per year

WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME = 12  # Spouse working
WIFE_SUPPLEMENTAL_INCOME = 30_000  # per year
ME_YEARS_WITH_SUPPLEMENTAL_INCOME = 0  # Set to 0 by default; adjust as needed
ME_SUPPLEMENTAL_INCOME = 0  # per year; set to 0 by default; adjust as needed

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
    last_prob_se = float("inf")

    while n_sims <= max_n_sims:
        simulation_data = run_simulation_mp(
            n_sims=n_sims,
            initial_balance=initial_balance,
            withdrawal=withdrawal,
            withdrawal_negative_year=withdrawal_negative_year,
            sp500_percentage=sp500_percentage,
            years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
            social_security_money=SOCIAL_SECURITY_MONEY,
            wife_years_with_supplemental_income=WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME,
            wife_supplemental_income=WIFE_SUPPLEMENTAL_INCOME,
            me_years_with_supplemental_income=ME_YEARS_WITH_SUPPLEMENTAL_INCOME,
            me_supplemental_income=ME_SUPPLEMENTAL_INCOME,
            **tuning_run_simulation_common_kwargs(),
        )

        _p_hat, prob_se, _succ = cumulative_success_metrics(simulation_data.final_balances)
        should_stop, _reason = prob_convergence_should_stop(
            n_done=n_sims,
            prob_se=prob_se,
            prev_prob_se=last_prob_se,
            min_sims=MIN_SIMS_FOR_CONVERGENCE,
            se_acceptance=PROB_SUCCESS_SE_ACCEPTANCE,
            se_diff_threshold=PROB_SUCCESS_SE_DIFF_THRESHOLD,
        )
        if should_stop:
            break

        last_prob_se = prob_se
        n_sims += step_n_sims

    # Store how many simulations were used for this trial
    trial.set_user_attr("n_sims_used", n_sims)

    # Already between 0 and 1
    prob_success = float(simulation_data.probability_of_success)

    # final_balances is always 1D array of ending balances
    final_balances = simulation_data.final_balances
    final_balance_growth_ratios = final_balances / initial_balance

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

    trials_df = trials_to_dataframe(study)
    best_trial = safe_best_trial(study)
    sum_n_sims_all_trials = sum(
        t.user_attrs.get("n_sims_used", 0) for t in study.trials
    )
    print(f"\nTotal simulations run across all trials: {sum_n_sims_all_trials:,}")
    validation_summary = None
    if best_trial is None:
        print(f"\n=== No completed trials in {STUDY_NAME} — skipping best-trial report ===")
    else:
        best_params = best_trial.params
        print(f"\n=== Optimization Results for {STUDY_NAME} ===")
        print(f"Best Trial #{best_trial.number}")
        print(f"  Initial Balance: ${best_params['initial_balance']:,}")
        print(f"  Withdrawal (total spending need): ${best_params['withdrawal']:,}")
        print(f"  Withdrawal Negative Year: ${best_params['withdrawal_negative_year']:,}")
        print(
            f"  Withdrawal ratio: {best_params['withdrawal_percentage_when_negative_year']:.2f}"
        )
        print(f"  Retirement years: {RETIREMENT_YEARS} years")
        print(f"  Sampling mode: {SAMPLING_MODE} (block size: {BLOCK_BOOTSTRAP_SIZE})")
        print(
            f"  SP500: {best_params['sp500_percentage']:.2%}, Bond rate: {BOND_RATE:.2%}, Inflation rate: {INFLATION_RATE:.2%}"
        )
        print(f"  Social Security starts after {YEARS_WITHOUT_SOCIAL_SECURITY} years, amount: ${SOCIAL_SECURITY_MONEY:,} per year")
        print(
            f"  Wife Supplemental Income for {WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME} "
            f"years, amount: ${WIFE_SUPPLEMENTAL_INCOME:,} per year"
        )
        print(
            f"  Me Supplemental Income for {ME_YEARS_WITH_SUPPLEMENTAL_INCOME} "
            f"years, amount: ${ME_SUPPLEMENTAL_INCOME:,} per year"
        )
        for line in format_withdrawal_breakdown(
            withdrawal=best_params["withdrawal"],
            withdrawal_negative_year=best_params["withdrawal_negative_year"],
            wife_supplemental_income=WIFE_SUPPLEMENTAL_INCOME,
            wife_years_with_supplemental_income=WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME,
            me_supplemental_income=ME_SUPPLEMENTAL_INCOME,
            me_years_with_supplemental_income=ME_YEARS_WITH_SUPPLEMENTAL_INCOME,
            social_security_money=SOCIAL_SECURITY_MONEY,
            years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
            n_years=RETIREMENT_YEARS,
        ):
            print(line)
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

        print("\n~~~~~Running final validation simulation~~~~~")
        final_data = run_simulation_mp(
            n_sims=best_trial.user_attrs["n_sims_used"],
            initial_balance=best_params["initial_balance"],
            withdrawal=best_params["withdrawal"],
            withdrawal_negative_year=best_params["withdrawal_negative_year"],
            sp500_percentage=best_params["sp500_percentage"],
            years_without_social_security=YEARS_WITHOUT_SOCIAL_SECURITY,
            social_security_money=SOCIAL_SECURITY_MONEY,
            wife_years_with_supplemental_income=WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME,
            wife_supplemental_income=WIFE_SUPPLEMENTAL_INCOME,
            me_years_with_supplemental_income=ME_YEARS_WITH_SUPPLEMENTAL_INCOME,
            me_supplemental_income=ME_SUPPLEMENTAL_INCOME,
            **tuning_run_simulation_common_kwargs(),
        )
        print(f"Final Probability of Success: {final_data.probability_of_success:.3%}")
        print(f"Success rate SE (binomial): {final_data.probability_of_success_se:.4%}")
        print(f"Standard Deviation (ending balance): ${final_data.std_final:,.0f}")
        print(f"Standard error of mean ending balance: ${final_data.std_error:,.0f}")
        mf = float(final_data.final_balances.mean())
        if mf != 0.0:
            print(f"Relative SE (mean ending balance): {final_data.std_error / mf:.3%}")

        final_data.print_stats()
        validation_summary = simulation_data_summary(final_data)

    print("\n=== End of optimization ===")
    end_time = time.perf_counter()
    wall_time_sec = end_time - start_time
    print(f"Total optimization time: {wall_time_sec:.2f} seconds")

    fixed_config = {
        "script": "04_tuning.py",
        "PROB_SUCCESS_SE_DIFF_THRESHOLD": PROB_SUCCESS_SE_DIFF_THRESHOLD,
        "PROB_SUCCESS_SE_ACCEPTANCE": PROB_SUCCESS_SE_ACCEPTANCE,
        "MIN_SIMS_FOR_CONVERGENCE": MIN_SIMS_FOR_CONVERGENCE,
        "SAMPLING_MODE": SAMPLING_MODE,
        "BLOCK_BOOTSTRAP_SIZE": BLOCK_BOOTSTRAP_SIZE,
        "INITIAL_BALANCE_RANGE": list(INITIAL_BALANCE_RANGE),
        "INITIAL_BALANCE_STEP": INITIAL_BALANCE_STEP,
        "WITHDRAWAL_RANGE": list(WITHDRAWAL_RANGE),
        "WITHDRAWAL_STEP": WITHDRAWAL_STEP,
        "WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE": list(
            WITHDRAWAL_NEGATIVE_YEAR_PERCENTAGE_RANGE
        ),
        "WITHDRAWAL_NEGATIVE_STEP": WITHDRAWAL_NEGATIVE_STEP,
        "N_SIMS_RANGE": list(N_SIMS_RANGE),
        "STEP_N_SIMS": STEP_N_SIMS,
        "TRIAL_COUNT": TRIAL_COUNT,
        "STORAGE_PATH": STORAGE_PATH,
        "STUDY_NAME": STUDY_NAME,
        "RETIREMENT_YEARS": RETIREMENT_YEARS,
        "PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN": PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MIN,
        "PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX": PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_MAX,
        "PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP": PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP,
        "INFLATION_RATE": INFLATION_RATE,
        "BOND_RATE": BOND_RATE,
        "YEARS_WITHOUT_SOCIAL_SECURITY": YEARS_WITHOUT_SOCIAL_SECURITY,
        "SOCIAL_SECURITY_MONEY": SOCIAL_SECURITY_MONEY,
        "WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME": WIFE_YEARS_WITH_SUPPLEMENTAL_INCOME,
        "WIFE_SUPPLEMENTAL_INCOME": WIFE_SUPPLEMENTAL_INCOME,
        "ME_YEARS_WITH_SUPPLEMENTAL_INCOME": ME_YEARS_WITH_SUPPLEMENTAL_INCOME,
        "ME_SUPPLEMENTAL_INCOME": ME_SUPPLEMENTAL_INCOME,
        "OBJECTIVE_WEIGHTS": {
            "prob_success": 0.50,
            "withdrawal": 0.15,
            "initial_balance": 0.10,
            "withdrawal_consistency": 0.10,
            "final_balance": 0.15,
        },
    }
    RUN_RESULTS = build_run_results(
        study_name=STUDY_NAME,
        storage_url=STORAGE_PATH,
        fixed_config=fixed_config,
        wall_time_sec=wall_time_sec,
        trials_df=trials_df,
        study=study,
        validation_summary=validation_summary,
    )
    export_run_results_sidecars(RUN_RESULTS, json_path=RESULTS_JSON_PATH)
