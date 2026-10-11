import json
import os
import pytest
from src.weather_client import decode_weather_code, WeatherApiError

def test_decode_weather_code():
    assert decode_weather_code(0) == "Clear sky"
    assert decode_weather_code(61) == "Slight rain"
    assert "Unknown" in decode_weather_code(999)

def test_offline_geocoding_parsing():
    sample_path = os.path.join(os.path.dirname(__file__), "sample_geocoding.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            data = json.load(f)
        assert "results" in data
        assert len(data["results"]) > 0
        assert "latitude" in data["results"][0]

def test_offline_forecast_parsing():
    sample_path = os.path.join(os.path.dirname(__file__), "sample_forecast.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            data = json.load(f)
        assert "daily" in data
        assert "temperature_2m_max" in data["daily"]