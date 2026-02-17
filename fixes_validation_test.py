"""
Tests to validate fixes made to the simulation logic.
"""
import numpy as np
import pytest
from common import run_simulation_mp
from random_utils import generate_constrained_indices


def test_random_seed_reproducibility():
    """Test that using the same random seed produces identical results."""
    params = {
        "n_sims": 1000,
        "n_years": 30,
        "initial_balance": 1_000_000,
        "withdrawal": 40_000,
        "withdrawal_negative_year": 30_000,
        "random_seed": 42,
    }

    # Run simulation twice with same seed
    result1 = run_simulation_mp(**params)
    result2 = run_simulation_mp(**params)

    # Results should be identical
    np.testing.assert_array_equal(result1.final_balances, result2.final_balances)
    assert result1.probability_of_success == result2.probability_of_success


def test_random_seed_different_results():
    """Test that different seeds produce different results."""
    params = {
        "n_sims": 1000,
        "n_years": 30,
        "initial_balance": 1_000_000,
        "withdrawal": 40_000,
        "withdrawal_negative_year": 30_000,
    }

    # Run simulation twice with different seeds
    result1 = run_simulation_mp(**params, random_seed=42)
    result2 = run_simulation_mp(**params, random_seed=123)

    # Results should be different
    assert not np.array_equal(result1.final_balances, result2.final_balances)


def test_input_validation_negative_n_sims():
    """Test that negative n_sims raises error."""
    with pytest.raises(ValueError, match="n_sims must be positive"):
        run_simulation_mp(n_sims=-100)


def test_input_validation_negative_n_years():
    """Test that negative n_years raises error."""
    with pytest.raises(ValueError, match="n_years must be positive"):
        run_simulation_mp(n_years=-10)


def test_input_validation_n_years_exceeds_data():
    """Test that n_years > available data raises error."""
    with pytest.raises(ValueError, match="cannot exceed available historical data"):
        run_simulation_mp(n_years=200)  # We have ~150 years of data


def test_input_validation_negative_balance():
    """Test that negative initial balance raises error."""
    with pytest.raises(ValueError, match="initial_balance must be positive"):
        run_simulation_mp(initial_balance=-1000)


def test_input_validation_sp500_percentage_out_of_range():
    """Test that sp500_percentage outside [0, 1] raises error."""
    with pytest.raises(ValueError, match="sp500_percentage must be between 0 and 1"):
        run_simulation_mp(sp500_percentage=1.5)

    with pytest.raises(ValueError, match="sp500_percentage must be between 0 and 1"):
        run_simulation_mp(sp500_percentage=-0.1)


def test_input_validation_withdrawal_negative_year_warning():
    """Test that withdrawal_negative_year > withdrawal raises warning."""
    with pytest.warns(UserWarning, match="withdrawing MORE in down markets"):
        run_simulation_mp(
            n_sims=100,
            n_years=10,
            withdrawal=50_000,
            withdrawal_negative_year=60_000,
        )


def test_constrained_indices_returns_correct_length():
    """Test that constrained sampling always returns exactly n_years indices."""
    rng = np.random.default_rng(42)

    # Create mock returns data
    returns = np.random.randn(150) * 0.1  # 150 years of ~10% std dev returns

    # Test various n_years values
    for n_years in [10, 20, 30, 40]:
        indices = generate_constrained_indices(rng, returns, n_years)
        assert len(indices) == n_years, f"Expected {n_years} indices, got {len(indices)}"
        assert indices.dtype == np.int64


def test_constrained_indices_input_validation():
    """Test that constrained sampling validates inputs."""
    rng = np.random.default_rng(42)
    returns = np.random.randn(100) * 0.1

    # Test negative n_years
    with pytest.raises(ValueError, match="n_years must be positive"):
        generate_constrained_indices(rng, returns, -5)

    # Test n_years > available data
    with pytest.raises(ValueError, match="cannot exceed available returns"):
        generate_constrained_indices(rng, returns, 200)


def test_constrained_indices_constraints_applied():
    """Test that constraints are actually applied to generated sequences."""
    rng = np.random.default_rng(42)

    # Create returns with mix of positive and negative
    returns = np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, -0.15, 0.25, -0.05] * 15)

    # Generate constrained sequence
    max_consec_neg = 3
    indices = generate_constrained_indices(
        rng, returns, n_years=40, max_consec_neg=max_consec_neg
    )

    # Check that we don't exceed consecutive negative constraint
    selected_returns = returns[indices]
    consecutive_neg_count = 0
    max_consecutive_neg_found = 0

    for r in selected_returns:
        if r < 0:
            consecutive_neg_count += 1
            max_consecutive_neg_found = max(max_consecutive_neg_found, consecutive_neg_count)
        else:
            consecutive_neg_count = 0

    # Allow some slack since constraints may be relaxed when stuck
    # but should mostly be respected
    assert max_consecutive_neg_found <= max_consec_neg + 2, \
        f"Found {max_consecutive_neg_found} consecutive negatives, max allowed was {max_consec_neg}"


def test_simulation_with_social_security():
    """Test that social security income is correctly applied."""
    params = {
        "n_sims": 100,
        "n_years": 30,
        "initial_balance": 1_000_000,
        "withdrawal": 50_000,
        "withdrawal_negative_year": 50_000,
        "years_without_social_security": 20,  # SS starts at year 20
        "social_security_money": 30_000,  # $30k/year SS
        "random_seed": 42,
    }

    result = run_simulation_mp(**params)

    # With SS, should have better outcomes than without
    result_no_ss = run_simulation_mp(**{**params, "social_security_money": 0})

    # Probability of success should be higher with social security
    assert result.probability_of_success >= result_no_ss.probability_of_success


def test_simulation_with_supplemental_income():
    """Test that supplemental income is correctly applied."""
    params = {
        "n_sims": 100,
        "n_years": 30,
        "initial_balance": 1_000_000,
        "withdrawal": 50_000,
        "withdrawal_negative_year": 50_000,
        "years_with_supplemental_income": 10,  # Extra income for first 10 years
        "supplemental_income": 20_000,  # $20k/year
        "random_seed": 42,
    }

    result = run_simulation_mp(**params)

    # With supplemental income, should have better outcomes
    result_no_supp = run_simulation_mp(**{**params, "supplemental_income": 0})

    # Probability of success should be higher with supplemental income
    assert result.probability_of_success >= result_no_supp.probability_of_success


def test_mixed_portfolio_allocation():
    """Test that mixed stock/bond allocation works correctly."""
    params = {
        "n_sims": 100,
        "n_years": 20,
        "initial_balance": 1_000_000,
        "withdrawal": 40_000,
        "withdrawal_negative_year": 40_000,
        "random_seed": 42,
    }

    # 100% stocks
    result_stocks = run_simulation_mp(**{**params, "sp500_percentage": 1.0})

    # 50/50 allocation
    result_mixed = run_simulation_mp(**{**params, "sp500_percentage": 0.5})

    # 100% bonds
    result_bonds = run_simulation_mp(**{**params, "sp500_percentage": 0.0, "bond_rate": 0.04})

    # All should complete without error and produce results
    assert result_stocks.n_sims == 100
    assert result_mixed.n_sims == 100
    assert result_bonds.n_sims == 100

    # Bonds only should have lower variance (more predictable)
    assert result_bonds.std_final < result_stocks.std_final


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
