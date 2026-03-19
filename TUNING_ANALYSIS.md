# Hyperparameter Optimization Analysis - 04_tuning.py

## Issues Found & Recommended Improvements

### 🔴 CRITICAL ISSUES

#### 1. **Missing Parameters in Final Validation** (Line 252-264)
**Severity**: HIGH - Results don't match optimization

**Problem**: The final validation simulation is missing parameters that were used during optimization:
```python
# During optimization (lines 72-86)
run_simulation_mp(
    ...
    years_with_supplemental_income=YEARS_WITH_SUPPLEMENTAL_INCOME,  # ✓ Used
    supplemental_income=SUPPLEMENTAL_INCOME,  # ✓ Used
)

# Final validation (lines 252-264)
run_simulation_mp(
    ...
    # ✗ MISSING: years_with_supplemental_income
    # ✗ MISSING: supplemental_income
)
```

**Impact**: Final validation shows different probability of success than the best trial.

**Fix**: Add missing parameters to final validation.

---

#### 2. **Redundant Parameter Creates Inefficient Search** (Lines 46-57)
**Severity**: MEDIUM - Wastes optimization budget

**Problem**: Two dependent parameters are suggested independently:
```python
withdrawal_percentage_when_negative_year = trial.suggest_float(...)  # Parameter 1
withdrawal_negative_year = trial.suggest_float(  # Parameter 2 (depends on 1)
    withdrawal * withdrawal_percentage_when_negative_year,
    withdrawal,
    step=WITHDRAWAL_STEP,
)
```

**Issue**: `withdrawal_negative_year` is completely determined by `withdrawal` and `withdrawal_percentage_when_negative_year`, so suggesting it as a separate parameter creates redundant search space. Optuna doesn't know these are dependent.

**Fix**: Only suggest `withdrawal_percentage_when_negative_year` and compute `withdrawal_negative_year` from it:
```python
withdrawal_percentage_when_negative_year = trial.suggest_float(...)
withdrawal_negative_year = withdrawal * withdrawal_percentage_when_negative_year
```

---

#### 3. **Random Noise Pollutes Optimization**
**Severity**: MEDIUM - Reduces optimization quality

**Problem**: Each trial uses different random seed, so trials aren't fairly comparable. A mediocre parameter set with lucky RNG can beat a good parameter set with unlucky RNG.

**Example**:
- Trial A: great params, unlucky markets → score 0.75
- Trial B: okay params, lucky markets → score 0.80
- Optuna thinks B is better, but it's just luck

**Fix**: Use fixed `random_seed` for all trials so only parameters vary:
```python
simulation_data = run_simulation_mp(
    ...
    random_seed=42,  # Same for all trials
)
```

---

### 🟡 OPTIMIZATION IMPROVEMENTS

#### 4. **Adaptive Simulation Wastes Computation**
**Severity**: MEDIUM - Inefficient use of resources

**Problem**: Adaptive simulation throws away previous runs:
```python
n_sims = 50_000  # First iteration
n_sims = 100_000  # Second iteration (re-runs all 100k, wastes the previous 50k)
n_sims = 150_000  # Third iteration (re-runs all 150k, wastes previous 100k)
```

**Impact**: If a trial needs 150k simulations, it runs 50k + 100k + 150k = 300k total (100% waste).

**Better Approach**: Progressive batching - accumulate results from each batch:
```python
n_sims_done = 0
all_balances = []

while n_sims_done < max_n_sims:
    # Run ONLY the next batch
    batch_size = min(step_n_sims, max_n_sims - n_sims_done)
    batch_results = run_simulation_mp(n_sims=batch_size, ...)
    all_balances.append(batch_results.final_balances)

    # Compute cumulative statistics
    cumulative_balances = np.concatenate(all_balances)
    std_error = np.std(cumulative_balances) / np.sqrt(len(cumulative_balances))

    # Check convergence
    if converged:
        break
```

**Benefit**: Only runs needed simulations, could save 50-66% of computation.

---

#### 5. **Probability Threshold Too Harsh** (Line 131)
**Severity**: LOW-MEDIUM - May eliminate good solutions

**Problem**:
```python
prob_term = exponential(prob_success, 0.99, 1.0, 7)
```

This maps probability of success to score:
- 99.0% → 0.00 (basically zero)
- 99.5% → 0.50
- 100% → 1.00

**Issue**: Solutions with 95% or 98% success rate get near-zero score, even if they have much higher withdrawals or better final balances. This might prematurely eliminate good tradeoffs.

**Better Approach**: Wider range that still rewards high probability:
```python
prob_term = exponential(prob_success, 0.90, 1.0, 5)
# 90% → 0.00
# 95% → 0.40
# 99% → 0.87
# 100% → 1.00
```

Or use a multi-threshold approach:
```python
if prob_success < 0.90:
    prob_term = 0.0  # Reject outright
else:
    prob_term = exponential(prob_success, 0.90, 1.0, 5)
```

---

#### 6. **Unused Variables Computed** (Lines 119-126)
**Severity**: LOW - Code cleanliness

**Problem**: `vmin` and `vmax` are calculated but never used in the objective:
```python
vmin = float(np.percentile(final_balance_growth_ratios, 5))
vmax = float(np.percentile(final_balance_growth_ratios, 95))
# ... validation logic ...
# But then these aren't used for anything except user_attrs
```

**Fix**: Either use them for normalization, or remove the calculation.

---

#### 7. **Potential Division by Zero** (Line 95)
**Severity**: LOW - Edge case

**Problem**: If all simulations fail (mean balance ≈ 0):
```python
std_error_relative_to_mean = std_error / simulation_data.final_balances.mean()
```

**Fix**: Add safety check:
```python
mean_balance = simulation_data.final_balances.mean()
if mean_balance > 0:
    std_error_relative_to_mean = std_error / mean_balance
else:
    std_error_relative_to_mean = float('inf')  # Force more simulations
```

---

#### 8. **Single-Threaded Optimization** (Line 191)
**Severity**: MEDIUM - Performance

**Problem**:
```python
study.optimize(objective, n_trials=TRIAL_COUNT, n_jobs=1)
```

**Issue**: Only runs one trial at a time. With `n_jobs > 1`, multiple trials can run in parallel.

**Consideration**: Each trial already uses multiprocessing internally. Need to balance:
- `n_jobs=1`: Sequential trials, each trial uses all cores
- `n_jobs=4`: 4 parallel trials, each trial gets fewer cores

**Recommendation**:
```python
# If you have 16 cores and want to run 2 trials in parallel:
n_jobs = 2
# Each trial will automatically use ~8 cores for its simulation
```

---

#### 9. **No Pruning Strategy**
**Severity**: LOW-MEDIUM - Missed optimization opportunity

**Problem**: Optuna supports pruning (stopping unpromising trials early), but it's not used.

**Potential**: With adaptive simulation, could prune after 50k simulations if probability of success is already < 80%, saving the additional 50k-400k simulations.

**Implementation**:
```python
# After each simulation batch
trial.report(prob_success, step=n_sims)
if trial.should_prune():
    raise optuna.TrialPruned()
```

---

### 🟢 ENHANCEMENT OPPORTUNITIES

#### 10. **Hard-Coded Income Parameters**
**Severity**: LOW - Missed optimization opportunity

Currently hard-coded:
- `YEARS_WITHOUT_SOCIAL_SECURITY = 20`
- `SOCIAL_SECURITY_MONEY = 50_000`
- `YEARS_WITH_SUPPLEMENTAL_INCOME = 12`
- `SUPPLEMENTAL_INCOME = 30_000`

**Opportunity**: These could be optimization parameters if you want to explore:
- "Should I delay retirement if I can work part-time for 5 more years?"
- "How much does delaying social security by 2 years help?"

**Consideration**: Adds 4 more dimensions to search space, may need more trials.

---

#### 11. **Objective Function Weights**
**Severity**: LOW - Tuning opportunity

Current weights:
```python
score = (
    prob_term * 0.50          # Probability of success
    + withdrawal_term * 0.15  # Higher withdrawals
    + initial_balance_term * 0.10  # Lower initial balance
    + withdrawal_diff_ratio_term * 0.10  # Consistent withdrawals
    + final_balance_term * 0.15  # Better final balance
)
```

**Questions**:
1. Is 50% weight on probability too high? Might force conservative solutions.
2. Should "lower initial balance" and "higher withdrawal" compete more directly?
3. Is "consistent withdrawals" worth 10%? This encourages same withdrawal in good/bad years.

**Recommendation**: Could make weights configurable constants or even use multi-objective optimization.

---

#### 12. **Parameter Discretization**
**Severity**: LOW - Search efficiency

Current steps:
- `INITIAL_BALANCE_STEP = 200_000` → 31 possible values
- `WITHDRAWAL_STEP = 5_000` → 12 possible values
- `PERCENTAGE_INVESTMENT_IN_STOCKS_VS_BOND_STEP = 0.05` → 7 possible values

**Total search space**: 31 × 12 × 7 × ... ≈ tens of thousands of combinations

**Consideration**:
- Steps might be too coarse (missing sweet spots)
- Or too fine (wasting trials on trivial differences)

Example: Is there really a meaningful difference between $4.2M and $4.4M initial balance?

**Recommendation**: Analyze sensitivity to step size.

---

## Summary of Recommendations

### Must Fix (Breaking Bugs)
1. ✅ Add missing supplemental income parameters to final validation
2. ✅ Remove redundant `withdrawal_negative_year` parameter suggestion
3. ✅ Add division-by-zero protection

### High Impact Improvements
4. ✅ Add fixed `random_seed` for fair trial comparison
5. ✅ Implement progressive batching for adaptive simulation
6. ⚠️ Widen probability threshold range (0.90-1.0 instead of 0.99-1.0)

### Medium Impact Improvements
7. ⚠️ Consider parallel trials with `n_jobs > 1`
8. ⚠️ Implement pruning for early trial termination
9. ⚠️ Remove or use `vmin`/`vmax` calculation

### Nice to Have
10. ⚠️ Make income parameters tunable
11. ⚠️ Make objective weights configurable
12. ⚠️ Analyze parameter discretization

---

## Implementation Priority

**Phase 1: Critical Fixes** (Do now)
- Fix missing parameters in final validation
- Remove redundant parameter
- Add safety checks (division by zero)
- Add random seed for reproducibility

**Phase 2: Performance** (Do if optimization is slow)
- Progressive batching
- Parallel trials
- Pruning strategy

**Phase 3: Tuning** (Do if results aren't satisfactory)
- Adjust probability threshold
- Review objective weights
- Experiment with parameter ranges

**Phase 4: Advanced** (Do if exploring options)
- Make income parameters tunable
- Multi-objective optimization
