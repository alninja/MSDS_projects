"""
Domain models, forecasting methods, and analytical functions for UBOS District Population Forecaster.
"""

from abc import ABC, abstractmethod
import statistics
import numpy as np
import pandas as pd


class DistrictPopulation:
    """Represents a district's historical population data over consecutive years."""

    def __init__(self, name: str, years: list[int] | np.ndarray, populations: list[float] | np.ndarray):
        """
        Initialize a DistrictPopulation object.

        Parameters
        ----------
        name : str
            Name of the district.
        years : list or np.ndarray
            Array of historical years.
        populations : list or np.ndarray
            Array of population values (e.g., in thousands).
        """
        self.name = name
        self.years = np.array(years, dtype=int)
        self.populations = np.array(populations, dtype=float)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validates input lengths, non-negativity, and year ordering."""
        if len(self.years) != len(self.populations):
            raise ValueError(f"Years ({len(self.years)}) and Populations ({len(self.populations)}) length mismatch.")
        if len(self.years) == 0:
            raise ValueError("Input series cannot be empty.")
        if np.any(self.populations < 0):
            raise ValueError("Population values cannot be negative.")
        if not np.all(np.diff(self.years) == 1):
            raise ValueError("Years array must be strictly sequential (consecutive).")

    def stats_statistics(self) -> dict[str, float]:
        """Compute statistics using Python's statistics module."""
        pops = self.populations.tolist()
        return {
            "mean": statistics.mean(pops),
            "median": statistics.median(pops),
            "variance": statistics.variance(pops),
            "stdev": statistics.stdev(pops),
        }

    def stats_numpy(self, ddof: int = 1) -> dict[str, float]:
        """Compute statistics using NumPy (default ddof=1 for sample statistics)."""
        return {
            "mean": float(np.mean(self.populations)),
            "median": float(np.median(self.populations)),
            "variance": float(np.var(self.populations, ddof=ddof)),
            "stdev": float(np.std(self.populations, ddof=ddof)),
        }

    def compute_yoy_growth(self) -> np.ndarray:
        """Compute Year-on-Year growth rates as decimals (e.g., 0.05 for 5%)."""
        return np.diff(self.populations) / self.populations[:-1]

    def compute_cagr(self) -> float:
        """Compute Compound Annual Growth Rate (CAGR)."""
        n_periods = len(self.years) - 1
        if n_periods <= 0:
            return 0.0
        return float((self.populations[-1] / self.populations[0]) ** (1 / n_periods) - 1)

    def __len__(self) -> int:
        return len(self.populations)

    def __repr__(self) -> str:
        return f"DistrictPopulation(name='{self.name}', start_year={self.years[0]}, end_year={self.years[-1]}, count={len(self)})"


# =====================================================================
# Forecasting Models (OOP Structure)
# =====================================================================

class Forecaster(ABC):
    """Abstract Base Class for time series forecasting models."""

    def __init__(self):
        self.fitted = False
        self.last_year = None
        self.last_val = None

    @abstractmethod
    def fit(self, years: np.ndarray, populations: np.ndarray) -> "Forecaster":
        """Fit the model to historical training data."""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast 'horizon' steps into the future."""
        pass


class LinearTrendForecaster(Forecaster):
    """Linear trend forecaster using np.polyfit degree 1."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> "LinearTrendForecaster":
        self.years_train = np.array(years)
        self.slope, self.intercept = np.polyfit(years, populations, deg=1)
        self.last_year = years[-1]
        self.last_val = populations[-1]
        self.fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        future_years = np.arange(self.last_year + 1, self.last_year + 1 + horizon)
        return self.slope * future_years + self.intercept


class ExponentialCAGRForecaster(Forecaster):
    """Exponential forecaster based on Compound Annual Growth Rate (CAGR)."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> "ExponentialCAGRForecaster":
        n_periods = len(years) - 1
        self.cagr = (populations[-1] / populations[0]) ** (1 / n_periods) - 1
        self.last_year = years[-1]
        self.last_val = populations[-1]
        self.fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        steps = np.arange(1, horizon + 1)
        return self.last_val * ((1 + self.cagr) ** steps)


class FibonacciRatioForecaster(Forecaster):
    """
    Forecaster scaling the last historical value by successive Fibonacci ratios F(n)/F(n-1).
    F_1 = 1, F_2 = 1, F_3 = 2, F_4 = 3, F_5 = 5, F_6 = 8, F_7 = 13 ...
    For forecast step k = 1..horizon, scaling multiplier = F(k+2) / F(k+1).
    """

    @staticmethod
    def _get_fibonacci(n: int) -> list[int]:
        fibs = [1, 1]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs

    def fit(self, years: np.ndarray, populations: np.ndarray) -> "FibonacciRatioForecaster":
        self.last_year = years[-1]
        self.last_val = populations[-1]
        self.fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        # Need up to horizon + 2 Fibonacci numbers
        fibs = self._get_fibonacci(horizon + 3)
        forecasts = []
        for k in range(1, horizon + 1):
            ratio = fibs[k + 2] / fibs[k + 1]
            forecasts.append(self.last_val * ratio)
        return np.array(forecasts)


# =====================================================================
# Error Metrics & Helper Functions
# =====================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE metrics."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def validate_models(
    years: np.ndarray, pops: np.ndarray, split_year: int = 2021
) -> tuple[pd.DataFrame, str]:
    """
    Trains forecasters on years <= split_year, tests on years > split_year.
    Returns comparison table and name of best model based on MAE.
    """
    train_idx = years <= split_year
    test_idx = years > split_year

    train_years, test_years = years[train_idx], years[test_idx]
    train_pops, test_pops = pops[train_idx], pops[test_idx]

    models = {
        "Linear Trend": LinearTrendForecaster(),
        "Exponential CAGR": ExponentialCAGRForecaster(),
        "Fibonacci Ratio": FibonacciRatioForecaster(),
    }

    results = []
    for name, model in models.items():
        model.fit(train_years, train_pops)
        preds = model.predict(len(test_years))
        metrics = compute_metrics(test_pops, preds)
        metrics["Model"] = name
        results.append(metrics)

    df_res = pd.DataFrame(results)[["Model", "MAE", "RMSE", "MAPE"]]
    best_model_name = df_res.sort_values("MAE").iloc[0]["Model"]
    return df_res, best_model_name


def estimate_classrooms(population_increase_thousands: float, age_pct: float = 0.18, capacity: int = 53) -> dict:
    """
    Estimates required primary classrooms for a given population increase (in thousands).
    """
    pop_increase = population_increase_thousands * 1000
    pupil_increase = pop_increase * age_pct
    classrooms_needed = int(np.ceil(pupil_increase / capacity))
    return {
        "pop_increase": pop_increase,
        "pupil_increase": pupil_increase,
        "classrooms_needed": classrooms_needed,
    }


def bootstrap_residuals_pi(
    model_cls, years: np.ndarray, pops: np.ndarray, horizon: int = 5, n_bootstraps: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap residual resampling to calculate 95% prediction intervals.
    """
    rng = np.random.default_rng(seed)
    model = model_cls().fit(years, pops)
    in_sample_preds = model.predict(len(years) - 1) if isinstance(model, Forecaster) else None
    
    # Generate fitted values manually for in-sample residual calculation
    if isinstance(model, LinearTrendForecaster):
        fitted = model.slope * years + model.intercept
    elif isinstance(model, ExponentialCAGRForecaster):
        steps = np.arange(len(years))
        fitted = pops[0] * ((1 + model.cagr) ** steps)
    else:  # Fibonacci
        fitted = np.full_like(pops, pops[-1])

    residuals = pops - fitted
    point_forecast = model.predict(horizon)

    boot_forecasts = np.zeros((n_bootstraps, horizon))
    for b in range(n_bootstraps):
        resamp_resids = rng.choice(residuals, size=horizon, replace=True)
        boot_forecasts[b, :] = point_forecast + resamp_resids

    lower_bound = np.percentile(boot_forecasts, 2.5, axis=0)
    upper_bound = np.percentile(boot_forecasts, 97.5, axis=0)

    return point_forecast, lower_bound, upper_bound