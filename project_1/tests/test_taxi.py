"""
tests/test_taxi.py
Unit tests for Mini-Project 5 using pytest.
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.taxi import (
    Route,
    solve_ntinda_equilibrium,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    LinearTrendForecaster,
    backtest_walk_forward,
    plan_fleet_size,
)


def test_route_validation():
    """Test validation errors for empty arrays, negative passengers, and invalid fares."""
    with pytest.raises(ValueError, match="cannot be empty"):
        Route("Empty", [], fare_ugx=2000)

    with pytest.raises(ValueError, match="cannot be negative"):
        Route("Negative", [30, -5, 40], fare_ugx=2000)

    with pytest.raises(ValueError, match="strictly positive"):
        Route("Zero Fare", [30, 40], fare_ugx=0)


def test_equilibrium_solver():
    """Test equilibrium solution for Ntinda route."""
    res = solve_ntinda_equilibrium(fare_current=2000.0)
    assert res["equilibrium_price_ugx"] == 2200.0
    assert res["equilibrium_quantity"] == 76.0
    assert res["fare_status"] == "Below Equilibrium"


def test_forecasters_single_step():
    """Test single step ahead predictions across forecasters."""
    series = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    ma = MovingAverageForecaster(window=3)
    assert np.isclose(ma.fit_predict(series), 40.0)

    lt = LinearTrendForecaster()
    assert np.isclose(lt.fit_predict(series), 60.0)

    ses = ExponentialSmoothingForecaster(alpha=0.5)
    assert ses.fit_predict(series) > 0.0


def test_fleet_planning():
    """Test fleet size estimation and buffer rounding."""
    res = plan_fleet_size(forecasted_passengers=250.0, buffer_pct=0.15)
    # 250 / 112 = 2.232. With 15% buffer = 2.56 -> ceil = 3 vehicles
    assert res["deployed_vehicles"] == 3