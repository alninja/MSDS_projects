"""
Domain models, equilibirum solver, time series forecasters, 
rolling-origin backtesting and fleetplanning logic for mini-project 5.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any 
import statistics 
import numpy as np 
import pandas as pd 
import scipy.linalg as la

class Route:
    """
    Represents a taxi route with daily passenger counts and fare per trip.
    """

    def __init__(self, name:str, daily_passengers: List[float] | np.ndarray, fare_ugx: float):
        """
        Parameters
        --------------------------
        name : str 
            Route name (e.g Kampala-Luzira)
        daily_passengers : list or np.ndarray
            Array of daily passenger counts.
        fare_ugx : float 
            Fare per passenger in  UGX.
        """
        self.name = name
        self.daily_passengers = np.array(daily_passengers, dtype=float)
        self.fare_ugx = float(fare_ugx)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validates array length, fare and passenger non-negativity."""
        if len(self.daily_passengers) == 0:
            raise ValueError("Passenger counts array cannot be empty.")
        if np.any(self.daily_passengers < 0):
            raise ValueError("Passenger counts cannot be negative.")
        if self.fare_ugx <= 0:
            raise ValueError("Fare must be strictly positive.")

    def daily_revenue(self) -> np.ndarray:
        # calculates the daily revenue in UGX (Passengers * Fare)
        return self.daily_passengers * self.fare_ugx

    def total_revenue(self) -> float:
        # calculates total cumulative revenue across all days
        return float(np.sum(self.daily_revenue))

    def passenger_stats(self) -> Dict[str, float]:
        # Calculates descriptive statistics using python' statistics module. 
        pass_list = self.daily_passengers.tolist()
        return {
            "mean": statistics.mean(pass_list),
            "median": statistics.median(pass_list),
            "variance": statistics.variance(pass_list),
            "stdev": statistics.stdev(pass_list),

        }

    def __len__(self) -> int :
        # Customising the len function
        return len(self.daily_passengers)

    def __repr__(self):
        return f"Route(name='{self.name}', days={len(self)}, fare_ugx={self.fare_ugx:,.0f})"


#########################################
# Supply demand equilibrium solver
#########################################

def solve_ntinda_equilibrium(fare_current: float = 2000.0) -> Dict[str, Any]:
    """
    Solves demand supply equilibrium for Ntinda route:
    Qd = 120 - 0.02*P => Q + 0.02*P = 120
    Qs = 10 + 0.03*P => Q - 0.03*P = 10 

    System matrix A * [Q, P]^T = [120, 10]^T
    """ 
    A = np.array([
        [1.0, 0.02],
        [1.0, -0.03]
    ], dtype=float)
    b = np.array([120.0, 10.0], dtype=float)

    sol = la.solve(A, b)
    q_eq, p_eq = float(sol[0]), float(sol[1])

    is_below = fare_current < p_eq
    status = "Below Equilibrium" if is_below else ("Above Equilibrium" if fare_current > p_eq else "At Equilibrium")
    return {
        "equilibrium_quantity":q_eq,
        "equilibrium_price_ugx": p_eq,
        "current_fare_ugx": fare_current,
        "fare_status": status
    }

#########################################################
# Time series forecaster
#########################################################

class Forecaster(ABC):
    """Abstract base class for revenue and demand forecasting models"""
    @abstractmethod
    def fit_predict(self, series: np.ndarray) -> float:
        """ Fits on historical series and predicts the next step ahead"""
        pass

class MovingAverageForecaster(Forecaster):
    """ Simple N-day Moving Average Forecaster. """

    def __init__(self,window: int = 3):
        self.window = window 

    def fit_predict(self, series: np.ndarray) -> float:
        if len(series) < self.window:
            return float(np.mean(series))
        return float(np.mean(series[-self.window:]))

class ExponentialSmoothingForecaster(Forecaster):
    """Simple Exponential Smoothing (SES) with tunable alpha (0 < alpha <= 1)."""

    def __init__(self, alpha: float = 0.3):
        if not (0 < alpha <= 1.0):
            raise ValueError("Alpha parameter must lie in range (0, 1].")
        self.alpha = alpha

    def fit_predict(self, series: np.ndarray) -> float:
        s = series[0]
        for val in series[1:]:
            s = self.alpha * val + (1 - self.alpha) * s
        return float(s)


class LinearTrendForecaster(Forecaster):
    """Linear trend forecaster using np.polyfit degree 1 over time indices."""

    def fit_predict(self, series: np.ndarray) -> float:
        n = len(series)
        x = np.arange(n)
        slope, intercept = np.polyfit(x, series, deg=1)
        return float(slope * n + intercept)


class SeasonalNaiveForecaster(Forecaster):
    """Extension: Seasonal-Naïve forecaster using lag period (e.g. 7 for weekly pattern)."""

    def __init__(self, season_length: int = 7):
        self.season_length = season_length

    def fit_predict(self, series: np.ndarray) -> float:
        if len(series) < self.season_length:
            return float(series[-1])
        return float(series[-self.season_length])


# =====================================================================
# Backtesting, Fleet Planning & Simulation
# =====================================================================

def backtest_walk_forward(series: np.ndarray, model: Forecaster, start_day: int = 3) -> float:
    """
    Evaluates model performance using rolling-origin walk-forward backtesting.
    Returns Mean Absolute Error (MAE).
    """
    errors = []
    for t in range(start_day, len(series)):
        train_data = series[:t]
        actual = series[t]
        pred = model.fit_predict(train_data)
        errors.append(abs(actual - pred))
    return float(np.mean(errors))


def tune_ses_alpha(series: np.ndarray, alpha_grid: List[float] = None) -> Tuple[float, float]:
    """Grid searches optimal alpha for Simple Exponential Smoothing."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.95, 19).tolist()

    best_alpha = alpha_grid[0]
    best_mae = float("inf")

    for a in alpha_grid:
        model = ExponentialSmoothingForecaster(alpha=a)
        mae = backtest_walk_forward(series, model)
        if mae < best_mae:
            best_mae = mae
            best_alpha = a

    return best_alpha, best_mae


def plan_fleet_size(
    forecasted_passengers: float, 
    trips_per_vehicle: int = 8, 
    vehicle_capacity: int = 14, 
    buffer_pct: float = 0.15
) -> Dict[str, Any]:
    """
    Calculates vehicle requirements for Day 11 given demand forecasts.
    """
    daily_capacity_per_vehicle = trips_per_vehicle * vehicle_capacity  # 112 passengers/day
    raw_vehicles = forecasted_passengers / daily_capacity_per_vehicle
    vehicles_with_buffer = raw_vehicles * (1.0 + buffer_pct)
    deployed_vehicles = int(np.ceil(vehicles_with_buffer))

    return {
        "forecasted_passengers": round(forecasted_passengers, 1),
        "raw_vehicles_needed": round(raw_vehicles, 2),
        "buffer_pct": buffer_pct,
        "deployed_vehicles": deployed_vehicles
    }


def simulate_60day_route(seed: int = 2026) -> np.ndarray:
    """
    Generates 60 days of passenger demand with weekly seasonality (busier Fridays, quieter Sundays).
    """
    rng = np.random.default_rng(seed)
    days = np.arange(60)
    
    # Seasonality multiplier: Mon-Thu=1.0, Fri=1.3, Sat=1.1, Sun=0.7
    day_of_week = days % 7
    seasonal_multipliers = np.array([1.0, 1.0, 1.0, 1.0, 1.3, 1.1, 0.7])
    multipliers = seasonal_multipliers[day_of_week]

    base_demand = 50.0
    trend = 0.1 * days
    noise = rng.normal(0, 4.0, size=60)

    simulated_demand = (base_demand + trend) * multipliers + noise
    return np.clip(simulated_demand, 10.0, None)