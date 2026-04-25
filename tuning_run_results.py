"""Helpers to aggregate Optuna tuning runs into DataFrames and structured dicts."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from common import SimulationData


def _trial_state_counts(study: optuna.Study) -> dict[str, int]:
    return dict(Counter(t.state.name for t in study.trials))


def safe_best_trial(study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
    try:
        return study.best_trial
    except ValueError:
        return None


def trials_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective_value": float(trial.value) if trial.value is not None else np.nan,
        }
        if trial.duration is not None:
            row["duration_seconds"] = trial.duration.total_seconds()
        else:
            row["duration_seconds"] = np.nan
        for k, v in trial.params.items():
            row[f"params_{k}"] = v
        for k, v in trial.user_attrs.items():
            row[f"attr_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def simulation_data_summary(sd: SimulationData) -> dict[str, Any]:
    mean_final = float(np.mean(sd.final_balances))
    median_final = float(np.median(sd.final_balances))
    out: dict[str, Any] = {
        "n_sims": int(sd.n_sims),
        "probability_of_success": float(sd.probability_of_success),
        "probability_of_success_se": float(sd.probability_of_success_se),
        "wilson_success_95": [float(sd.wilson_success_95[0]), float(sd.wilson_success_95[1])],
        "wilson_success_99": [float(sd.wilson_success_99[0]), float(sd.wilson_success_99[1])],
        "std_final": float(sd.std_final),
        "std_error": float(sd.std_error),
        "mean_final_balance": mean_final,
        "median_final_balance": median_final,
        "initial_balance": float(sd.initial_balance),
        "withdrawal": float(sd.withdrawal),
        "withdrawal_negative_year": float(sd.withdrawal_negative_year),
        "n_years": int(sd.n_years),
        "go_back_year": int(sd.go_back_year),
        "annual_expense_ratio": float(sd.annual_expense_ratio),
        "sampling_mode": sd.sampling_mode,
        "bond_return_mode": sd.bond_return_mode,
        "inflation_rate": sd.inflation_rate,
    }
    if sd.reserve_floor is not None and sd.probability_below_reserve_floor is not None:
        out["reserve_floor"] = float(sd.reserve_floor)
        out["probability_below_reserve_floor"] = float(sd.probability_below_reserve_floor)
    return out


def build_run_results(
    *,
    study_name: str,
    storage_url: str,
    fixed_config: dict[str, Any],
    wall_time_sec: float,
    trials_df: pd.DataFrame,
    study: optuna.Study,
    validation_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    state_counts = _trial_state_counts(study)
    total_n_sims = sum(t.user_attrs.get("n_sims_used", 0) for t in study.trials)
    attr_col = "attr_n_sims_used"
    if attr_col in trials_df.columns:
        total_from_df = int(pd.to_numeric(trials_df[attr_col], errors="coerce").fillna(0).sum())
    else:
        total_from_df = 0

    best: Optional[dict[str, Any]] = None
    bt = safe_best_trial(study)
    if bt is not None:
        best = {
            "trial_number": bt.number,
            "params": dict(bt.params),
            "user_attrs": dict(bt.user_attrs),
            "value": bt.value,
        }

    return {
        "study_name": study_name,
        "storage_url": storage_url,
        "fixed_config": fixed_config,
        "wall_time_sec": wall_time_sec,
        "trials_df": trials_df,
        "aggregates": {
            "total_trials": len(study.trials),
            "state_counts": state_counts,
            "complete": state_counts.get(TrialState.COMPLETE.name, 0),
            "pruned": state_counts.get(TrialState.PRUNED.name, 0),
            "fail": state_counts.get(TrialState.FAIL.name, 0),
            "total_n_sims": total_n_sims,
            "total_n_sims_from_df": total_from_df,
        },
        "best": best,
        "validation": validation_summary,
    }


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def export_run_results_sidecars(
    run_results: dict[str, Any],
    *,
    json_path: str | None = None,
) -> None:
    if json_path is not None:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        payload = {
            "study_name": run_results["study_name"],
            "storage_url": run_results["storage_url"],
            "fixed_config": _json_safe(run_results["fixed_config"]),
            "wall_time_sec": run_results["wall_time_sec"],
            "aggregates": _json_safe(run_results["aggregates"]),
            "best": _json_safe(run_results["best"]),
            "validation": _json_safe(run_results["validation"]),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
