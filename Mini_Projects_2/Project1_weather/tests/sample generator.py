import json
import requests
import os 

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Fetch and save sample geocoding response
geo_res = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": "Dresden", "count": 1})
geo_path = os.path.join(current_dir, "sample_geocoding.json")
with open(geo_path, "w", encoding="utf-8") as f:
    json.dump(geo_res.json(), f, indent=2)

# 2. Fetch and save sample forecast response (using Dresden coordinates: lat 51.05, lon 13.74)
forecast_res = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": 51.05,
    "longitude": 13.74,
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "weathercode"],
    "timezone": "auto",
    "forecast_days": 3
})

forecast_path = os.path.join(current_dir, "sample_forecast.json")
with open(forecast_path, "w", encoding="utf-8") as f:
    json.dump(forecast_res.json(), f, indent=2)

print("Sample JSON files successfully generated and saved to tests/")