"""
tests/test_population.py
Unit tests for Mini-Project 1 using pytest.
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.population import (
    DistrictPopulation,
    LinearTrendForecaster,
    ExponentialCAGRForecaster,
    FibonacciRatioForecaster,
    estimate_classrooms,
)


def test_district_population_validation_mismatch():
    """Test length mismatch error raised."""
    with pytest.raises(ValueError, match="length mismatch"):
        DistrictPopulation("Test", years=[2020, 2021], populations=[100])


def test_district_population_negative_values():
    """Test edge case: negative population raised."""
    with pytest.raises(ValueError, match="negative"):
        DistrictPopulation("Test", years=[2020, 2021], populations=[100, -50])


def test_district_population_empty_input():
    """Test edge case: empty input raised."""
    with pytest.raises(ValueError, match="empty"):
        DistrictPopulation("Test", years=[], populations=[])


def test_stats_variance_ddof_comparison():
    """Verify statistics vs numpy sample variance equality."""
    dp = DistrictPopulation("Test", years=[2020, 2021, 2022], populations=[100, 200, 300])
    s_stats = dp.stats_statistics()
    np_stats = dp.stats_numpy(ddof=1)
    assert np.isclose(s_stats["variance"], np_stats["variance"])


def test_forecasters_prediction():
    """Test standard execution of forecasters."""
    years = np.array([2015, 2016, 2017, 2018, 2019])
    pops = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

    # Linear
    lf = LinearTrendForecaster().fit(years, pops)
    l_pred = lf.predict(2)
    assert len(l_pred) == 2
    assert np.isclose(l_pred[0], 150.0)

    # CAGR
    ef = ExponentialCAGRForecaster().fit(years, pops)
    e_pred = ef.predict(2)
    assert len(e_pred) == 2
    assert e_pred[0] > 140.0

    # Fibonacci
    ff = FibonacciRatioForecaster().fit(years, pops)
    f_pred = ff.predict(2)
    assert len(f_pred) == 2


def test_classroom_estimation():
    """Test classroom requirements logic."""
    res = estimate_classrooms(population_increase_thousands=10.0, age_pct=0.18, capacity=53)
    # 10,000 * 0.18 = 1800 pupils. 1800 / 53 = 33.96 -> 34 classrooms
    assert res["classrooms_needed"] == 34