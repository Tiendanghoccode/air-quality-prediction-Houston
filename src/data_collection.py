"""
data_collection.py
Collects raw air quality data for Houston from public APIs or local sources.
"""

import os
import requests
import pandas as pd


RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch_aqi_data(api_key: str, zip_code: str = "77001", distance: int = 25) -> dict:
    """Fetch AQI data from AirNow API for a given ZIP code."""
    url = "https://www.airnowapi.org/aq/observation/zipCode/current/"
    params = {
        "format": "application/json",
        "zipCode": zip_code,
        "distance": distance,
        "API_KEY": api_key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_raw_data(data: list, filename: str = "aqi_raw.csv") -> str:
    """Save raw data records to a CSV file in the raw data directory."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df = pd.DataFrame(data)
    output_path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_collection.py <API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]
    data = fetch_aqi_data(api_key)
    save_raw_data(data)
