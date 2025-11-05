import optuna
from common import run_simulation_mp

# Tuning & stopping constants
STD_ERROR_THRESHOLD = 0.05
STD_ERROR_ACCEPTANCE = 0.10

# Adaptive sim params
START_N_SIMS = 500_000
MAX_N_SIMS = 10_000_000
STEP_N_SIMS = 500_000

# How close to best probability to consider "tie" (e.g. within 0.5% absolute)
EPS_PROB = 0.005  # 0.5% = 0.005 in 0-1 scale


def objective(trial):
    initial_balance = trial.suggest_float(
        "initial_balance", 3_000_000, 8_000_000, step=250_000
    )

    n_sims = START_N_SIMS
    last_std_error = float("inf")

    while n_sims <= MAX_N_SIMS:
        simulation_data = run_simulation_mp(
            n_sims=n_sims,
            initial_balance=initial_balance,
            random_with_real_life_constraints=False,
        )

        std_error = simulation_data.std_error
        diff_percentage = (
            (last_std_error - std_error) / last_std_error
            if last_std_error != float("inf")
            else float("inf")
        )

        print(
            f"Trial {trial.number}: n_sims={n_sims:,}, std_error={std_error:.6f}, diff={diff_percentage:.4f}"
        )

        if std_error <= STD_ERROR_ACCEPTANCE or diff_percentage <= STD_ERROR_THRESHOLD:
            break

        last_std_error = std_error
        n_sims += STEP_N_SIMS

    # store metadata
    trial.set_user_attr("n_sims_used", n_sims)

    # probability as the single objective (0-1)
    prob_success = simulation_data.probability_of_success
    trial.set_user_attr("prob_success", prob_success)
    trial.set_user_attr("std_final", simulation_data.std_final)
    trial.set_user_attr("std_error", simulation_data.std_error)

    # Return prob_success for Optuna to maximize
    return prob_success


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, n_jobs=1)

    # best probability found (value is prob_success)
    best_prob = study.best_value
    print(f"Best observed probability (optuna best_value): {best_prob:.6f}")

    # Find trials within EPS_PROB of best_prob, pick one with lowest initial_balance
    candidate_trials = [
        t
        for t in study.trials
        if t.value is not None and (best_prob - t.value) <= EPS_PROB
    ]
    if not candidate_trials:
        # fallback: use best_trial
        chosen = study.best_trial
    else:
        chosen = min(candidate_trials, key=lambda t: t.params["initial_balance"])

    chosen_balance = chosen.params["initial_balance"]
    chosen_n_sims = chosen.user_attrs.get("n_sims_used", None)
    chosen_prob = chosen.user_attrs.get("prob_success", chosen.value)

    print("=== Selected trial (lexicographic): ===")
    print("Initial balance:", f"{chosen_balance:,}")
    print("Probability:", chosen_prob)
    print("n_sims used:", chosen_n_sims)

    # Final verify run if desired
    final_data = run_simulation_mp(n_sims=chosen_n_sims, initial_balance=chosen_balance)
    print("Final verify Probability:", final_data.probability_of_success)
    print("Final verify Std_error:", final_data.std_error)

    # Leaderboard: top 20 by probability, tie-breaker: lower balance
    print("\nTop 20 leaderboard (by prob desc, balance asc tiebreak):")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: (
            t.value,
            -t.params["initial_balance"],
        ),  # value desc, balance asc by negation
        reverse=True,
    )[:20]

    for i, t in enumerate(sorted_trials, 1):
        print(
            f"#{i} prob={t.value:.6f}, balance={t.params['initial_balance']:,}, n_sims={t.user_attrs.get('n_sims_used','N/A')}"
        )
