"""
tests/test_microgrid.py
Unit tests for Mini-Project 2 using pytest.
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.microgrid import MicroGrid, HybridMicroGrid, generate_30day_demand


def test_microgrid_solver_normal():
    """Test standard 2x2 solver output against hand calculation."""
    mg = MicroGrid()
    # 3x + 2y = 120, 4x + y = 110 -> x = 20, y = 30
    x, y = mg.solve_day(120.0, 110.0)
    assert np.isclose(x, 20.0)
    assert np.isclose(y, 30.0)


def test_microgrid_vectorised_solver():
    """Test equivalence between iterative and vectorised solvers."""
    mg = MicroGrid()
    demands = generate_30day_demand(seed=42)
    
    vec_sol = mg.solve_vectorised(demands)
    
    for i in range(30):
        x, y = mg.solve_day(demands[0, i], demands[1, i])
        assert np.isclose(vec_sol[0, i], x)
        assert np.isclose(vec_sol[1, i], y)


def test_edge_case_negative_input():
    """Test edge case handling for negative demand inputs."""
    mg = MicroGrid()
    with pytest.raises(ValueError, match="cannot be negative"):
        mg.solve_day(-50.0, 100.0)


def test_singular_matrix_rejection():
    """Test error raised when coefficient matrix is singular."""
    singular_A = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(ValueError, match="singular"):
        MicroGrid(coeff_matrix=singular_A)


def test_infeasibility_nnls():
    """Test Non-Negative Least Squares (NNLS) fallback on extreme inputs."""
    mg = MicroGrid()
    # D1 = 10, D2 = 200 leads to negative x in standard linear solve
    x, y, was_infeasible = mg.handle_infeasibility_nnls(10.0, 200.0)
    assert was_infeasible is True
    assert x >= 0.0
    assert y >= 0.0