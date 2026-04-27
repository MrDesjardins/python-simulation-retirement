"""
Shared scenario parameters for cross-script comparability.

Scripts should import from here instead of duplicating inflation, bond,
sampling, and horizon constants when they are meant to describe the same
economic world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

BondReturnMode = Literal["fixed", "historical"]
SamplingMode = Literal["random", "constrained", "block_bootstrap"]


# --- Optuna studies: one nominal world for 04_* and 05_* ---
TUNING_RETIREMENT_YEARS = 40
TUNING_INFLATION_RATE = 0.036
TUNING_BOND_RATE = 0.036
TUNING_SAMPLING_MODE: SamplingMode = "block_bootstrap"
TUNING_BLOCK_BOOTSTRAP_SIZE = 5
TUNING_BOND_RETURN_MODE: BondReturnMode = "fixed"

# Optional: report P(ending balance < floor) in print_stats / exports.
TUNING_DEFAULT_RESERVE_FLOOR_USD: Optional[float] = 500_000
# Advisory + fund expense drag applied to portfolio return each year (0 = off).
TUNING_DEFAULT_ANNUAL_EXPENSE_RATIO = 0.0


@dataclass(frozen=True)
class BaselineSuccessScriptConfig:
    """Canonical parameters for `01_success.py` and matching historical checks."""

    n_years: int = 45
    initial_balance: float = 4_000_000
    sampling_mode: SamplingMode = "block_bootstrap"
    block_bootstrap_size: int = 5
    sp500_percentage: float = 0.70
    bond_rate: float = 0.03
    bond_return_mode: BondReturnMode = "historical"
    inflation_rate: float = 0.03
    withdrawal: float = 110_000
    withdrawal_negative_year: float = 95_000
    social_security_money: float = 48_000
    years_without_social_security: int = 25
    wife_years_with_supplemental_income: int = 14
    wife_supplemental_income: float = 24_000
    me_years_with_supplemental_income: int = 0
    me_supplemental_income: float = 0
    annual_expense_ratio: float = 0.0
    reserve_floor: Optional[float] = 250_000

    def run_simulation_mp_kwargs(self) -> dict[str, Any]:
        return {
            "n_years": self.n_years,
            "initial_balance": self.initial_balance,
            "sampling_mode": self.sampling_mode,
            "block_bootstrap_size": self.block_bootstrap_size,
            "withdrawal": self.withdrawal,
            "withdrawal_negative_year": self.withdrawal_negative_year,
            "sp500_percentage": self.sp500_percentage,
            "bond_rate": self.bond_rate,
            "bond_return_mode": self.bond_return_mode,
            "inflation_rate": self.inflation_rate,
            "social_security_money": self.social_security_money,
            "years_without_social_security": self.years_without_social_security,
            "wife_years_with_supplemental_income": self.wife_years_with_supplemental_income,
            "wife_supplemental_income": self.wife_supplemental_income,
            "me_years_with_supplemental_income": self.me_years_with_supplemental_income,
            "me_supplemental_income": self.me_supplemental_income,
            "annual_expense_ratio": self.annual_expense_ratio,
            "reserve_floor": self.reserve_floor,
        }

    def format_withdrawal_breakdown_kwargs(self) -> dict[str, Any]:
        return {
            "withdrawal": self.withdrawal,
            "withdrawal_negative_year": self.withdrawal_negative_year,
            "wife_supplemental_income": self.wife_supplemental_income,
            "wife_years_with_supplemental_income": self.wife_years_with_supplemental_income,
            "me_supplemental_income": self.me_supplemental_income,
            "me_years_with_supplemental_income": self.me_years_with_supplemental_income,
            "social_security_money": self.social_security_money,
            "years_without_social_security": self.years_without_social_security,
            "n_years": self.n_years,
        }

    def run_simulation_historical_real_kwargs(self) -> dict[str, Any]:
        """Kwargs aligned with `run_simulation_mp` economics for rolling-window history."""
        return {
            "n_years": self.n_years,
            "initial_balance": self.initial_balance,
            "withdrawal": self.withdrawal,
            "withdrawal_negative_year": self.withdrawal_negative_year,
            "inflation_rate": self.inflation_rate,
            "sp500_percentage": self.sp500_percentage,
            "bond_rate": self.bond_rate,
            "bond_return_mode": self.bond_return_mode,
            "years_without_social_security": self.years_without_social_security,
            "social_security_money": self.social_security_money,
            "wife_years_with_supplemental_income": self.wife_years_with_supplemental_income,
            "wife_supplemental_income": self.wife_supplemental_income,
            "me_years_with_supplemental_income": self.me_years_with_supplemental_income,
            "me_supplemental_income": self.me_supplemental_income,
            "annual_expense_ratio": self.annual_expense_ratio,
            "reserve_floor": self.reserve_floor,
        }


BASELINE_SUCCESS_SCRIPT = BaselineSuccessScriptConfig()


def tuning_run_simulation_common_kwargs() -> dict[str, Any]:
    """Shared economics for Optuna objectives (income is script-specific)."""
    return {
        "n_years": TUNING_RETIREMENT_YEARS,
        "sampling_mode": TUNING_SAMPLING_MODE,
        "block_bootstrap_size": TUNING_BLOCK_BOOTSTRAP_SIZE,
        "bond_return_mode": TUNING_BOND_RETURN_MODE,
        "bond_rate": TUNING_BOND_RATE,
        "inflation_rate": TUNING_INFLATION_RATE,
        "annual_expense_ratio": TUNING_DEFAULT_ANNUAL_EXPENSE_RATIO,
        "reserve_floor": TUNING_DEFAULT_RESERVE_FLOOR_USD,
    }


def format_config_lines(title: str, items: dict[str, Any]) -> list[str]:
    """Human-readable lines for printing at script start."""
    lines = [title]
    for k, v in items.items():
        lines.append(f"  {k}: {v}")
    return lines
