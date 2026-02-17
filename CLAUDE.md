# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Monte Carlo retirement portfolio simulation tool that models portfolio sustainability over time. It uses historical S&P 500 data from Shiller to run probabilistic simulations with configurable parameters like initial balance, withdrawal rates, asset allocation, and income sources.

## Data Requirements

The simulation requires historical market data from Shiller:
- Download from: https://shillerdata.com/
- File location: `data/ie_data.xls`
- The Excel file contains S&P 500 real total return prices used to calculate annual returns

## Development Commands

### Environment Setup
```sh
uv init
```

### Running Simulations
```sh
# Basic success probability simulation (10M simulations)
uv run 01_success.py

# Trajectory visualization (plots individual simulation paths)
uv run 02_simulation_lines.py

# Historical rolling window analysis
uv run 03_historical_value.py
```

### Hyperparameter Optimization
```sh
# Run Optuna optimization trials
uv run 04_tuning.py

# Launch dashboard to visualize optimization results
uv run optuna-dashboard sqlite:///db.sqlite3
```

### Testing
```sh
# Run all unit tests
uv run pytest -v -s ./*_test.py
```

## Architecture

### Core Simulation Engine (`common.py`)

**Key Functions:**
- `run_simulation_mp()`: Multiprocessing-based Monte Carlo simulation engine
  - Uses worker pools to parallelize simulations across CPU cores
  - Supports adaptive portfolio allocation between stocks and bonds
  - Handles inflation adjustments, social security income, and supplemental income
  - Returns `SimulationData` object with probability of success and trajectory data

- `run_simulation_historical_real()`: Rolling window historical backtesting
  - Tests portfolio survival using actual historical sequences
  - Identifies which historical starting years would have failed

**Worker Pattern:**
- Global variables (`_RETURNS`, `_INITIAL_BALANCE`, etc.) are set via `_init_worker()` in each subprocess to avoid serialization overhead
- `_simulate_chunk()` performs vectorized numpy operations for efficient batch simulation
- Results are concatenated from all worker chunks

**Portfolio Mechanics:**
- Linear allocation: `portfolio_growth = 1 + sp500_frac * return + bond_frac * bond_rate`
- Withdrawal logic: Full withdrawal in positive years, reduced withdrawal in negative years
- Income sources: Social security (starts after N years), supplemental income (limited duration)
- Safety mechanism: Reset balance to initial if it drops below initial balance within first `go_back_year` years

### Random Constraints (`random_utils.py`)

`generate_constrained_indices()` creates realistic return sequences by limiting:
- Maximum consecutive negative years (default: 4)
- Cumulative decline during negative streaks (default: -85%)
- Maximum consecutive positive years (default: 9)
- Cumulative rise during positive streaks (default: +132%)

This prevents unrealistic "perfect storm" or "perfect boom" scenarios that pure random sampling might generate.

### Hyperparameter Optimization (`04_tuning.py`)

Uses Optuna to optimize retirement parameters:

**Objective Function:**
Composite score balancing multiple goals:
- Probability of success (50% weight): Exponential mapping emphasizes 0.99-1.0 range
- Higher withdrawals (15% weight): Maximize sustainable spending
- Lower initial balance (10% weight): Minimize required starting capital
- Smaller withdrawal reduction in negative years (10% weight): Maintain consistent lifestyle
- Median ending balance growth (15% weight): Preserve wealth for heirs/longevity

**Adaptive Simulation:**
Starts with 50K simulations, increases by 50K steps until:
- Standard error ≤ 0.5% of mean, OR
- Standard error improvement < 1% between steps, OR
- Reaches 500K simulations maximum

**Important Constants:**
- Update `STUDY_NAME` version number when changing constants or objective logic
- Current optimization uses SQLite storage for persistence and dashboard visualization

### Simulation Scripts

**01_success.py**: High-volume simulation (10M runs) for precise probability estimation with histogram visualization

**02_simulation_lines.py**: Trajectory plotting (1M runs) showing individual paths and percentile bands

**03_historical_value.py**: Historical backtesting using rolling windows through actual market data

## Key Configuration Parameters

When running or modifying simulations, important parameters include:

- `n_years`: Retirement duration (typically 35-45 years)
- `initial_balance`: Starting portfolio value
- `withdrawal` / `withdrawal_negative_year`: Annual spending (can differ by market conditions)
- `sp500_percentage`: Stock allocation (0.0-1.0, remainder is bonds)
- `bond_rate`: Expected bond return (e.g., 0.03 for 3%)
- `inflation_rate`: Annual inflation assumption (e.g., 0.03 for 3%)
- `random_with_real_life_constraints`: Toggle constrained vs unconstrained random sampling
- `social_security_money` / `years_without_social_security`: Delayed income source
- `supplemental_income` / `years_with_supplemental_income`: Early retirement income (e.g., part-time work)

## Data Flow

1. Load historical returns from `data/ie_data.xls` (Shiller data)
2. Calculate annual real total returns from price series
3. For each simulation:
   - Sample N years of returns (with or without constraints)
   - Apply portfolio allocation between stocks/bonds
   - Deduct inflation-adjusted withdrawals
   - Add social security and supplemental income when applicable
   - Track balance trajectory
4. Aggregate results into success probability, percentiles, and statistics

## Utility Functions

`common.py` includes mathematical transformation functions used in optimization:

- `exponential()`: Maps values to 0-1 range with rapid growth curve (rewards high values)
- `inverse_exponential()`: Maps values to 0-1 range with rapid decay (penalizes high values)
- `threshold_power_map()`: Amplifies values above threshold while zeroing those below

These create non-linear objective terms that emphasize critical ranges (e.g., probability of success 0.99-1.0).
