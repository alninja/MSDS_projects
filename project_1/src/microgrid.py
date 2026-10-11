"""
src/microgrid.py
Domain models, solvers, vectorised routines, and cost accounting for Mini-Project 2.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
import scipy.linalg as la
from scipy.optimize import nnls


class MicroGrid:
    """
    Models a 2x2 Solar Micro-Grid dispatch system:
      3x + 2y = D1  (Daytime load, kWh)
      4x +  y = D2  (Critical-equipment load, kWh)
    """

    COEFF_MATRIX = np.array([
        [3.0, 2.0],
        [4.0, 1.0]
    ], dtype=float)

    SOLAR_COST_PER_KWH = 150.0   # UGX/kWh
    BATTERY_COST_PER_KWH = 450.0 # UGX/kWh

    def __init__(self, coeff_matrix: np.ndarray = None):
        """Initialize MicroGrid with coefficient matrix A."""
        self.A = np.array(coeff_matrix if coeff_matrix is not None else self.COEFF_MATRIX, dtype=float)
        self._validate_matrix()

    def _validate_matrix(self) -> None:
        """Validates matrix shape and checks for non-singularity."""
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"Coefficient matrix must be square, got shape {self.A.shape}.")
        if np.isclose(np.linalg.det(self.A), 0.0):
            raise ValueError("Coefficient matrix is singular (determinant is zero).")

    def get_system_diagnostics(self) -> Dict[str, float]:
        """Calculates determinant and condition number of the system matrix."""
        det_val = float(np.linalg.det(self.A))
        cond_num = float(np.linalg.cond(self.A))
        return {"determinant": det_val, "condition_number": cond_num}

    def solve_day(self, d1: float, d2: float) -> Tuple[float, float]:
        """
        Solves dispatch for a single day using scipy.linalg.solve.
        Returns (x, y) = (solar, battery).
        """
        if d1 < 0 or d2 < 0:
            raise ValueError("Demand values cannot be negative.")
        b = np.array([d1, d2], dtype=float)
        sol = la.solve(self.A, b)
        return float(sol[0]), float(sol[1])

    def solve_vectorised(self, demand_matrix: np.ndarray) -> np.ndarray:
        """
        Solves system for multiple days in a single vectorised call.
        
        Parameters
        ----------
        demand_matrix : np.ndarray
            A (2, N) array where row 0 is D1 and row 1 is D2.

        Returns
        -------
        np.ndarray
            A (2, N) array where row 0 is solar (x) and row 1 is battery (y).
        """
        if demand_matrix.ndim != 2 or demand_matrix.shape[0] != 2:
            raise ValueError("Demand matrix must have shape (2, N).")
        if np.any(demand_matrix < 0):
            raise ValueError("Demand values cannot be negative.")
        return la.solve(self.A, demand_matrix)

    @classmethod
    def handle_infeasibility_nnls(cls, d1: float, d2: float) -> Tuple[float, float, bool]:
        """
        Handles physical infeasibility (negative generation) by solving
        a Non-Negative Least Squares (NNLS) problem.
        Returns (solar, battery, was_infeasible).
        """
        b = np.array([d1, d2], dtype=float)
        sol, _ = nnls(cls.COEFF_MATRIX, b)
        
        # Check standard linear solve to detect if it was infeasible
        std_sol = la.solve(cls.COEFF_MATRIX, b)
        was_infeasible = bool(np.any(std_sol < 0))
        
        return float(sol[0]), float(sol[1]), was_infeasible

    @classmethod
    def calculate_cost(cls, solar_kwh: Union[float, np.ndarray], battery_kwh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates daily dispatch cost in UGX."""
        return solar_kwh * cls.SOLAR_COST_PER_KWH + battery_kwh * cls.BATTERY_COST_PER_KWH


class HybridMicroGrid(MicroGrid):
    """
    Extension: 3x3 System adding a Diesel Generator (z).
      3x + 2y + 1.5z = D1
      4x + 1y + 1.0z = D2
      1x + 1y + 2.0z = D3  (Nighttime/Backup load, kWh)
    """

    HYBRID_COEFF_MATRIX = np.array([
        [3.0, 2.0, 1.5],
        [4.0, 1.0, 1.0],
        [1.0, 1.0, 2.0]
    ], dtype=float)

    DIESEL_COST_PER_KWH = 1200.0 # UGX/kWh

    def __init__(self, coeff_matrix: np.ndarray = None):
        matrix = coeff_matrix if coeff_matrix is not None else self.HYBRID_COEFF_MATRIX
        super().__init__(coeff_matrix=matrix)

    def solve_day_3x3(self, d1: float, d2: float, d3: float) -> Tuple[float, float, float]:
        """Solves 3x3 dispatch system returning (solar, battery, diesel)."""
        if min(d1, d2, d3) < 0:
            raise ValueError("Demand values cannot be negative.")
        b = np.array([d1, d2, d3], dtype=float)
        sol = la.solve(self.A, b)
        return float(sol[0]), float(sol[1]), float(sol[2])


# =====================================================================
# Utilities & Data Generators
# =====================================================================

def generate_30day_demand(seed: int = 2026) -> np.ndarray:
    """
    Generates realistic 30-day demand matrix with weekly seasonality and noise.
    Returns array of shape (2, 30).
    """
    rng = np.random.default_rng(seed)
    days = np.arange(30)
    
    # Weekly seasonality factor (higher load on weekdays, lower on weekends)
    day_of_week = days % 7
    weekly_pattern = np.where(day_of_week < 5, 1.15, 0.85)

    base_d1 = 350.0 * weekly_pattern + rng.normal(0, 15, size=30)
    base_d2 = 280.0 * weekly_pattern + rng.normal(0, 10, size=30)

    # Force a couple of extreme edge-case days to test physical infeasibility handling
    base_d1[5] = 50.0   # Extremely low D1 relative to D2
    base_d2[5] = 400.0

    return np.vstack([np.clip(base_d1, 10, None), np.clip(base_d2, 10, None)])


def validate_interactive_input(prompt_text: str) -> float:
    """Helper for robust interactive input validation."""
    while True:
        try:
            val_str = input(prompt_text).strip()
            if not val_str:
                print("Error: Input cannot be empty. Please enter a positive number.")
                continue
            val = float(val_str)
            if val < 0:
                print("Error: Demand cannot be negative. Please try again.")
                continue
            return val
        except ValueError:
            print("Error: Invalid numeric input. Please enter a valid number.")