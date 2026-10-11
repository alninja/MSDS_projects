import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
import urllib.request
import requests

class WeatherApiError(Exception):
    """Custom exception raised for unrecoverable API or network errors."""
    pass

def robust_api_call(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0, max_retries: int = 3) -> Dict[str, Any]:
    """
    Wrapper that sets a timeout, calls raise_for_status(), and retries on failures 
    (connection errors, 5xx responses) with exponential backoff and a retry cap.
    """
    backoff_factor = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                raise WeatherApiError(f"Connection/Timeout failure after {max_retries} attempts: {e}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if 500 <= status_code < 600:
                if attempt == max_retries:
                    raise WeatherApiError(f"Server error {status_code} after {max_retries} attempts: {e}")
            else:
                # Client errors (4xx) should not be retried
                raise WeatherApiError(f"Client HTTP error {status_code}: {e}")
        except Exception as e:
            raise WeatherApiError(f"Unexpected error during API call: {e}")
        
        # Exponential backoff sleep
        sleep_time = backoff_factor * (2 ** (attempt - 1))
        time.sleep(sleep_time)
    
    raise WeatherApiError("Max retries exceeded without successful response.")

def geocode_urllib(city_name: str, email: str) -> Dict[str, Any]:
    """Geocode a city name using only urllib.request, urllib.parse, and json."""
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    query_string = urlencode({"name": city_name, "count": 5})
    url = f"{base_url}?{query_string}"
    headers = {"User-Agent": f"StudentProject contact:{email}"}
    
    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            elapsed = time.time() - start_time
            status_code = response.status
            content_type = response.headers.get("Content-Type")
            data = json.loads(response.read().decode("utf-8"))
            return {
                "status_code": status_code,
                "content_type": content_type,
                "response_time": elapsed,
                "data": data
            }
    except Exception as e:
        raise WeatherApiError(f"urllib geocoding failed: {e}")

def geocode_requests(city_name: str, email: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Geocode using requests module with params and timeout."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 5}
    headers = {"User-Agent": f"StudentProject contact:{email}"}
    
    start_time = time.time()
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    elapsed = time.time() - start_time
    response.raise_for_status()
    
    return {
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "response_time": elapsed,
        "data": response.json()
    }

def get_forecast_data(lat: float, lon: float, days: int = 7, email: str = "") -> Dict[str, Any]:
    """Request daily max/min temp, precipitation sum, and max wind speed from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "weathercode"],
        "timezone": "auto",
        "forecast_days": days
    }
    headers = {"User-Agent": f"StudentProject contact:{email}"}
    return robust_api_call(url, params=params, headers=headers)

# WMO Weather interpretation code mapping
WMO_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snowfall", 73: "Moderate snowfall",
    75: "Heavy snowfall", 77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers",   86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 97: "Heavy thunderstorm",
    99: "Thunderstorm with heavy hail"
}

def decode_weather_code(code: int) -> str:
    return WMO_WEATHER_CODES.get(code, f"Unknown code ({code})")