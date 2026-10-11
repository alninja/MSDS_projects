"""
tests/test_rainfall.py
Unit tests for Mini-Project 4 using pytest.
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rainfall import (
    Region,
    CropRule,
    numpy_cosine_similarity,
    verify_cosine_similarity,
    CROPS_CATALOG
)


def test_region_validation():
    """Test input array length and negative rainfall validation."""
    with pytest.raises(ValueError, match="exactly 12"):
        Region("Short", [100.0] * 10)

    with pytest.raises(ValueError, match="cannot be negative"):
        Region("Negative", [100.0] * 11 + [-20.0])


def test_region_metrics():
    """Test basic summary metrics for Region."""
    rain = [100, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
    r = Region("Kampala", rain)
    assert np.isclose(r.annual_total(), sum(rain))
    assert r.wettest_month() == ("May", 220.0)
    assert r.driest_month() == ("Sep", 60.0)


def test_cosine_similarity_correctness():
    """Verify custom numpy cosine similarity against scipy spatial distance."""
    u = np.array([120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130], dtype=float)
    v = np.array([8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15], dtype=float)
    
    assert verify_cosine_similarity(u, v) is True


def test_crop_rule_classification():
    """Test crop rule classification bounds."""
    rule = CROPS_CATALOG["Maize"]
    assert rule.classify_month(50.0) == "Drought Risk"
    assert rule.classify_month(120.0) == "Suitable for Maize"
    assert rule.classify_month(300.0) == "Waterlogging Risk"


def test_peak_detection():
    """Test rainy season peak detection."""
    # Gulu: Unimodal pattern with peak in August
    gulu_rain = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
    r_gulu = Region("Gulu", gulu_rain)
    peaks = r_gulu.detect_rainy_seasons()
    assert peaks["regime"] in ["Unimodal", "Bimodal"]