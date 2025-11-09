import pytest
from common import exponential, inverse_exponential, threshold_power_map

def test_exponential_value_under_min_value() -> None:
    result = exponential(1, 5, 10)
    assert result == 0

def test_exponential_min_value() -> None:
    result = exponential(0, 0, 1)
    assert result == 0

def test_exponential_max_value() -> None:
    result = exponential(1, 0, 1)
    assert result == 1

def test_exponential_low_value() -> None:
    result = exponential(0.1, 0, 1)
    assert result == pytest.approx(0.396, abs=1e-3)

def test_exponential_high_value() -> None:
    result = exponential(0.9, 0, 1)
    assert result == pytest.approx(0.996, abs=1e-3)

def test_inverse_exponential_min_value() -> None:
    result = inverse_exponential(0, 0, 1)
    assert result == 1

def test_inverse_exponential_max_value() -> None:
    result = inverse_exponential(1, 0, 1)
    assert result == 0

def test_inverse_exponential_low_value() -> None:
    result = inverse_exponential(0.1, 0, 1)
    assert result == pytest.approx(0.996, abs=1e-3)

def test_inverse_exponential_high_value() -> None:
    result = inverse_exponential(0.9, 0, 1)
    assert result == pytest.approx(0.396, abs=1e-3)

def test_exponential_scale_under_threshold() -> None:
    result = threshold_power_map(0.74, 0.75)
    assert result == 0

def test_exponential_scale_at_threshold() -> None:
    result = threshold_power_map(0.75, 0.75)
    assert result == pytest.approx(0.75, abs=1e-3)

def test_exponential_scale_above_threshold() -> None:
    result = threshold_power_map(0.9, 0.75)
    assert result == pytest.approx(0.944, abs=1e-3)

def test_exponential_scale_at_max() -> None:
    result = threshold_power_map(1, 0.75)
    assert result == pytest.approx(1, abs=1e-3)