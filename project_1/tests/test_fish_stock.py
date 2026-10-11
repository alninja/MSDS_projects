"""
tests/test_fish_stock.py
Unit tests for Mini-Project 3 using pytest.
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fish_stock import FishStock, PriceModel, RiskAssessor, generate_fibonacci_baseline


def test_fish_stock_normal_simulation():
    """Test discrete logistic simulation output length and values."""
    fs = FishStock(r=0.4, K=10000.0, N0=4000.0)
    stock = fs.simulate(weeks=52, h=0.10)
    assert len(stock) == 53
    assert np.all(stock >= 0)


def test_fish_stock_invalid_inputs():
    """Test edge cases with invalid input parameters."""
    with pytest.raises(ValueError, match="must be non-negative"):
        FishStock(r=-0.1, K=10000.0, N0=4000.0)

    fs = FishStock()
    with pytest.raises(ValueError, match="Harvest rate h must be between 0 and 1"):
        fs.simulate(weeks=52, h=1.5)


def test_price_model_bounds():
    """Test that price random walk stays strictly within defined bounds."""
    pm = PriceModel(start_price=12000.0, min_price=9000.0, max_price=16000.0)
    prices = pm.simulate_path(weeks=52, step_std=1000.0, seed=42)
    assert np.all(prices >= 9000.0)
    assert np.all(prices <= 16000.0)


def test_fibonacci_baseline():
    """Test generation of baseline Fibonacci sequence."""
    fibs = generate_fibonacci_baseline(15)
    assert len(fibs) == 15
    assert fibs == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


def test_risk_assessor_stats():
    """Test risk metrics computation."""
    rev = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    stats = RiskAssessor.compute_revenue_stats(rev)
    assert np.isclose(stats["mean"], 300.0)
    assert RiskAssessor.classify_risk(stats["cv"]) in ["Low Risk", "Moderate Risk", "High Risk"]