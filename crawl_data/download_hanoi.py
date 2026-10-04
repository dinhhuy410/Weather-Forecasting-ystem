import os
import time

import pandas as pd
import requests


CITY_NAME = "Hanoi"
LATITUDE = 21.0285
LONGITUDE = 105.8542
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Hanoi.csv")
API_URL = "https://archive-api.open-meteo.com/v1/archive"

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
    "weather_code",
    "is_day",
]


def download_hanoi() -> bool:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "timezone": "Asia/Bangkok",
        "models": "era5",
    }

    print("=" * 70)
    print("HANOI WEATHER DATA DOWNLOADER")
    print("=" * 70)
    print(f"City       : {CITY_NAME}")
    print(f"Start date : {START_DATE}")
    print(f"End date   : {END_DATE}")
    print(f"Output     : {OUTPUT_FILE}")

    for attempt in range(1, 6):
        print(f"\n[DOWNLOAD] Attempt {attempt}/5")
        try:
            response = requests.get(API_URL, params=params, timeout=300)
            if response.status_code == 429:
                wait_time = 30 * attempt
                print(f"[429] Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()
            if "hourly" not in data:
                print("[ERROR] API response does not contain hourly data.")
                return False

            df = pd.DataFrame(data["hourly"])
            df["city"] = CITY_NAME
            df["latitude"] = LATITUDE
            df["longitude"] = LONGITUDE
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").drop_duplicates(subset=["time"])
            df = df.reset_index(drop=True)
            df.to_csv(OUTPUT_FILE, index=False)

            print("\n[SUCCESS]")
            print(f"Rows    : {len(df):,}")
            print(f"Columns : {len(df.columns):,}")
            print(f"Time    : {df['time'].min()} -> {df['time'].max()}")
            print(f"File    : {OUTPUT_FILE}")
            return True
        except requests.exceptions.RequestException as error:
            print(f"[ERROR] Request failed: {error}")
            if attempt < 5:
                time.sleep(30 * attempt)
        except Exception as error:
            print(f"[ERROR] Unexpected error: {error}")
            return False

    print("[FAILED] Maximum retries reached.")
    return False


if __name__ == "__main__":
    download_hanoi()
