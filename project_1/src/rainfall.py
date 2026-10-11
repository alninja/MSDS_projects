"""
src/rainfall.py
Domain models, crop suitability classification, corrected vector similarity,
peak detection, and advisory generation for Mini-Project 4.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine
from scipy.signal import find_peaks


class Region:
    """
    Represents a geographical region's monthly rainfall data (12 months).
    """

    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def __init__(self, name: str, rainfall: List[float] | np.ndarray):
        """
        Parameters
        ----------
        name : str
            Name of the region.
        rainfall : list or np.ndarray
            Monthly rainfall values in mm (12 months).
        """
        self.name = name
        self.rainfall = np.array(rainfall, dtype=float)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validates rainfall array length and non-negativity."""
        if len(self.rainfall) != 12:
            raise ValueError(f"Rainfall array must contain exactly 12 monthly values, got {len(self.rainfall)}.")
        if np.any(self.rainfall < 0):
            raise ValueError("Rainfall values cannot be negative.")

    def annual_total(self) -> float:
        """Returns total annual rainfall in mm."""
        return float(np.sum(self.rainfall))

    def mean_monthly(self) -> float:
        """Returns mean monthly rainfall in mm."""
        return float(np.mean(self.rainfall))

    def wettest_month(self) -> Tuple[str, float]:
        """Returns (month_name, mm) for the wettest month."""
        idx = int(np.argmax(self.rainfall))
        return self.MONTH_NAMES[idx], float(self.rainfall[idx])

    def driest_month(self) -> Tuple[str, float]:
        """Returns (month_name, mm) for the driest month."""
        idx = int(np.argmin(self.rainfall))
        return self.MONTH_NAMES[idx], float(self.rainfall[idx])

    def coefficient_of_variation(self) -> float:
        """Calculates Coefficient of Variation (CV) of monthly rainfall."""
        mean_val = self.mean_monthly()
        if mean_val == 0:
            return 0.0
        return float(np.std(self.rainfall, ddof=1) / mean_val)

    def detect_rainy_seasons(self, min_height: float = 80.0, distance: int = 2) -> Dict[str, Any]:
        """
        Detects rainy seasons (peaks) using scipy.signal.find_peaks.
        Classifies climate regime as Unimodal, Bimodal, or Aseasonal/Dry.
        """
        peaks, _ = find_peaks(self.rainfall, height=min_height, distance=distance)
        peak_months = [self.MONTH_NAMES[p] for p in peaks]
        
        n_peaks = len(peaks)
        if n_peaks == 1:
            regime = "Unimodal"
        elif n_peaks == 2:
            regime = "Bimodal"
        elif n_peaks > 2:
            regime = "Multimodal"
        else:
            regime = "Aseasonal / Dry"

        return {
            "regime": regime,
            "peak_count": n_peaks,
            "peak_months": peak_months,
            "peak_indices": peaks
        }

    def __repr__(self) -> str:
        return f"Region(name='{self.name}', annual_total={self.annual_total():.1f}mm, CV={self.coefficient_of_variation():.2f})"


class CropRule:
    """
    Defines monthly rainfall thresholds for crop suitability based on FAO agronomic standards.
    """

    def __init__(self, crop_name: str, min_mm: float, optimal_max_mm: float, max_mm: float, citation: str = "FAO ECOCROP"):
        """
        Parameters
        ----------
        crop_name : str
            Name of the crop (e.g., Maize, Beans, Coffee).
        min_mm : float
            Minimum monthly water requirement (below this = Drought Risk).
        optimal_max_mm : float
            Upper bound for optimal growth (between min_mm and this = Suitable).
        max_mm : float
            Maximum threshold (above this = Waterlogging Risk).
        citation : str
            Cited source for thresholds.
        """
        self.crop_name = crop_name
        self.min_mm = min_mm
        self.optimal_max_mm = optimal_max_mm
        self.max_mm = max_mm
        self.citation = citation

    def classify_month(self, mm: float) -> str:
        """Classifies a single month's rainfall for this crop."""
        if mm < self.min_mm:
            return "Drought Risk"
        elif self.min_mm <= mm <= self.optimal_max_mm:
            return f"Suitable for {self.crop_name}"
        elif self.optimal_max_mm < mm <= self.max_mm:
            return "Moderate Risk (High Rain)"
        else:
            return "Waterlogging Risk"


# Standard Crops Dictionary with FAO Citations
CROPS_CATALOG = {
    "Beans": CropRule("Beans", min_mm=50.0, optimal_max_mm=120.0, max_mm=180.0, citation="FAO ECOCROP (Beans, 2020)"),
    "Maize": CropRule("Maize", min_mm=80.0, optimal_max_mm=180.0, max_mm=250.0, citation="FAO Irrigation & Drainage Paper 33"),
    "Coffee": CropRule("Coffee", min_mm=100.0, optimal_max_mm=220.0, max_mm=300.0, citation="Uganda Coffee Development Authority (UCDA)")
}


# =====================================================================
# Vector Similarity & Distance Functions
# =====================================================================

def numpy_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculates vector cosine similarity using NumPy dot product:
    cos_sim = (u . v) / (||u|| * ||v||)
    """
    dot_prod = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(dot_prod / (norm_u * norm_v))


def verify_cosine_similarity(u: np.ndarray, v: np.ndarray) -> bool:
    """
    Verifies NumPy cosine similarity against scipy.spatial.distance.cosine.
    Note: scipy distance returns (1 - similarity).
    """
    np_sim = numpy_cosine_similarity(u, v)
    scipy_dist = scipy_cosine(u, v)
    return np.isclose(np_sim, 1.0 - scipy_dist)


def compute_similarity_matrices(regions_dict: Dict[str, Region]) -> Dict[str, pd.DataFrame]:
    """
    Computes Cosine Similarity, Pearson Correlation, and Euclidean Distance matrices.
    """
    names = list(regions_dict.keys())
    n = len(names)

    cos_mat = np.zeros((n, n))
    pearson_mat = np.zeros((n, n))
    euc_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            u = regions_dict[names[i]].rainfall
            v = regions_dict[names[j]].rainfall
            
            cos_mat[i, j] = numpy_cosine_similarity(u, v)
            pearson_mat[i, j] = np.corrcoef(u, v)[0, 1]
            euc_mat[i, j] = np.linalg.norm(u - v)

    return {
        "Cosine Similarity": pd.DataFrame(cos_mat, index=names, columns=names),
        "Pearson Correlation": pd.DataFrame(pearson_mat, index=names, columns=names),
        "Euclidean Distance (mm)": pd.DataFrame(euc_mat, index=names, columns=names)
    }


def generate_farmer_advisory(region: Region, crop_rule: CropRule) -> str:
    """
    Generates a concise farmer advisory note (<= 200 words).
    """
    peaks = region.detect_rainy_seasons()
    suitable_months = []
    
    for idx, mm in enumerate(region.rainfall):
        status = crop_rule.classify_month(mm)
        if "Suitable" in status:
            suitable_months.append(Region.MONTH_NAMES[idx])

    s_months_str = ", ".join(suitable_months) if suitable_months else "None"
    p_months_str = ", ".join(peaks["peak_months"]) if peaks["peak_months"] else "None"

    advisory = (
        f"FARMER ADVISORY NOTE: {region.name.upper()} REGION\n"
        f"Target Crop: {crop_rule.crop_name} (Threshold: {crop_rule.min_mm}-{crop_rule.optimal_max_mm} mm/month)\n\n"
        f"1. Seasonal Profile: {region.name} exhibits a {peaks['regime']} rainfall pattern, with peak rainy months in {p_months_str}. "
        f"Annual cumulative rainfall reaches {region.annual_total():.0f} mm.\n"
        f"2. Planting Window: Optimal months for planting {crop_rule.crop_name} are {s_months_str}. "
        f"Plant at the immediate start of peak rain windows to ensure adequate soil moisture during early vegetative growth.\n"
        f"3. Risk Management: During dry months, utilize water harvesting and mulching to conserve moisture. "
        f"Avoid planting during peak deluge months to reduce seed washout and waterlogging risks."
    )
    return advisory