"""
src/fish_stock.py
Domain models, discrete logistic growth simulation, random walk pricing, 
and VaR risk analysis for Mini-Project 3.
"""

from typing import Tuple, Dict, Any, List
import statistics
import numpy as np
import pandas as pd


class FishStock:
    """
    Models fish population using discrete logistic growth with harvesting:
    N(t+1) = N(t) + r * N(t) * (1 - N(t) / K) - h * N(t)
    """

    def __init__(self, r: float = 0.4, K: float = 10000.0, N0: float = 4000.0):
        """
        Parameters
        ----------
        r : float
            Intrinsic growth rate.
        K : float
            Carrying capacity in tonnes.
        N0 : float
            Initial stock in tonnes.
        """
        if r < 0 or K <= 0 or N0 < 0:
            raise ValueError("Parameters r, K, and N0 must be non-negative, and K > 0.")
        self.r = float(r)
        self.K = float(K)
        self.N0 = float(N0)

    def simulate(self, weeks: int = 52, h: float = 0.10, closed_weeks: int = 0) -> np.ndarray:
        """
        Simulate fish stock trajectory over a number of weeks.
        
        Parameters
        ----------
        weeks : int
            Number of weeks to simulate.
        h : float
            Harvest rate (fraction of stock harvested each week).
        closed_weeks : int
            Number of weeks per year where harvesting is paused (h=0).
            
        Returns
        -------
        np.ndarray
            Array of stock values N(t) for t = 0..weeks.
        """
        if weeks <= 0:
            raise ValueError("Simulation weeks must be greater than 0.")
        if not (0 <= h <= 1):
            raise ValueError("Harvest rate h must be between 0 and 1.")

        stock = np.zeros(weeks + 1, dtype=float)
        stock[0] = self.N0

        for t in range(weeks):
            # Check if current week falls into closed season (last `closed_weeks` of every 52-week cycle)
            current_week_in_year = t % 52
            effective_h = 0.0 if current_week_in_year >= (52 - closed_weeks) else h

            N_t = stock[t]
            growth = self.r * N_t * (1.0 - N_t / self.K)
            harvest = effective_h * N_t
            N_next = N_t + growth - harvest
            stock[t + 1] = max(0.0, N_next)  # Extinction floor

        return stock

    @staticmethod
    def calculate_msy(r: float = 0.4, K: float = 10000.0) -> float:
        """Computes theoretical Maximum Sustainable Yield (MSY = r * K / 4)."""
        return (r * K) / 4.0


class PriceModel:
    """Simulates weekly price (UGX/kg) via a bounded random walk."""

    def __init__(self, start_price: float = 12000.0, min_price: float = 9000.0, max_price: float = 16000.0):
        if not (min_price <= start_price <= max_price):
            raise ValueError("Start price must lie within min_price and max_price.")
        self.start_price = float(start_price)
        self.min_price = float(min_price)
        self.max_price = float(max_price)

    def simulate_path(self, weeks: int = 52, step_std: float = 500.0, seed: int = None) -> np.ndarray:
        """Generates a bounded random walk price path."""
        rng = np.random.default_rng(seed)
        prices = np.zeros(weeks, dtype=float)
        prices[0] = self.start_price

        for t in range(1, weeks):
            change = rng.normal(0, step_std)
            new_price = prices[t - 1] + change
            prices[t] = np.clip(new_price, self.min_price, self.max_price)

        return prices


class RiskAssessor:
    """Evaluates revenue risk using Coefficient of Variation (CV) and Value-at-Risk (VaR)."""

    @staticmethod
    def compute_revenue_stats(revenue: np.ndarray) -> Dict[str, float]:
        """Calculates mean, median, variance, std dev, and CV of revenue."""
        rev_list = revenue.tolist()
        mean_val = statistics.mean(rev_list)
        std_val = statistics.stdev(rev_list)
        
        return {
            "mean": mean_val,
            "median": statistics.median(rev_list),
            "variance": statistics.variance(rev_list),
            "stdev": std_val,
            "cv": std_val / mean_val if mean_val > 0 else 0.0,
        }

    @staticmethod
    def classify_risk(cv: float) -> str:
        """Classifies revenue risk based on Coefficient of Variation."""
        if cv < 0.15:
            return "Low Risk"
        elif cv < 0.30:
            return "Moderate Risk"
        else:
            return "High Risk"

    @classmethod
    def run_monte_carlo_var(
        cls,
        stock_model: FishStock,
        price_model: PriceModel,
        weeks: int = 52,
        h: float = 0.10,
        n_sims: int = 1000,
        seed: int = 2026
    ) -> Tuple[np.ndarray, float, float]:
        """
        Runs Monte Carlo price paths to compute 5% Value-at-Risk (VaR) of annual revenue.
        
        Returns
        -------
        Tuple[np.ndarray, float, float]
            (annual_revenues, mean_annual_revenue, var_5_percentile)
        """
        rng = np.random.default_rng(seed)
        stock_path = stock_model.simulate(weeks=weeks, h=h)[:-1]  # Exclude t=52 stock value
        harvest_kg = (stock_path * h) * 1000.0  # Convert tonnes to kg

        annual_revenues = np.zeros(n_sims, dtype=float)

        for i in range(n_sims):
            # Seed each path deterministically from main generator
            path_seed = rng.integers(0, 1_000_000)
            prices = price_model.simulate_path(weeks=weeks, seed=path_seed)
            weekly_rev = harvest_kg * prices
            annual_revenues[i] = np.sum(weekly_rev)

        mean_rev = float(np.mean(annual_revenues))
        var_5 = float(np.percentile(annual_revenues, 5))
        return annual_revenues, mean_rev, var_5


def generate_fibonacci_baseline(n: int = 15) -> List[int]:
    """Generates first n terms of Fibonacci sequence."""
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:n]