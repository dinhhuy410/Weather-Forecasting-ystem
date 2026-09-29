import requests
import pandas as pd
import time
import os
# ============================================================
# CONFIGURATION
# ============================================================
CITY_NAME = "Hanoi"
LATITUDE = 21.0285
LONGITUDE = 105.8542
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Hanoi.csv")
# Open-Meteo Historical Weather API
API_URL = "https://archive-api.open-meteo.com/v1/archive"
# ============================================================
# WEATHER FEATURES
# ============================================================
VARIABLES = [
    # -------------------------
    # Atmospheric conditions
    # -------------------------
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    # -------------------------
    # Precipitation
    # -------------------------
    "precipitation",
    "rain",
    "snowfall",
    # -------------------------
    # Pressure
    # -------------------------
    "pressure_msl",
    "surface_pressure",
    # -------------------------
    # Cloud
    # -------------------------
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    # -------------------------
    # Wind
    # -------------------------
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    # -------------------------
    # Evaporation / humidity
    # -------------------------
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    # -------------------------
    # Soil temperature
    # -------------------------
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    # -------------------------
    # Soil moisture
    # -------------------------
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
    # ------------------------
    # Weather condition
    # -------------------------
    "weather_code",
    # -------------------------
    # Day / night
    # -------------------------
    "is_day",
]
# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================
# DOWNLOAD FUNCTION
# ============================================================
def download_hanoi():
    print("=" * 70)
    print("HANOI WEATHER DATA DOWNLOADER")
    print("=" * 70)

    print(f"City       : {CITY_NAME}")
    print(f"Latitude   : {LATITUDE}")
    print(f"Longitude  : {LONGITUDE}")
    print(f"Start date : {START_DATE}")
    print(f"End date   : {END_DATE}")
    print(f"Features   : {len(VARIABLES)}")
    print(f"Output     : {OUTPUT_FILE}")
    print("=" * 70)
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "timezone": "Asia/Bangkok",
        "models": "era5",
    }
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        print(f"\n[DOWNLOAD] Attempt "
            f"{attempt}/{max_retries}")
        try:
            response = requests.get(API_URL, params=params,timeout=300)
            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------
            if response.status_code == 429:
                wait_time = 30 * attempt
                print(f"[429] API rate limit.")
                print(f"Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            # ------------------------------------------------
            # HTTP errors
            # ------------------------------------------------
            response.raise_for_status()
            # ------------------------------------------------
            # Convert JSON
            # ------------------------------------------------
            data = response.json()
            if "hourly" not in data:
                print( "[ERROR] API response does "
                    "not contain hourly data.")
                return False
            # ------------------------------------------------
            # Create DataFrame
            # ------------------------------------------------
            df = pd.DataFrame(data["hourly"])
            # ------------------------------------------------
            # Add location information
            # ------------------------------------------------
            df["city"] = CITY_NAME
            df["latitude"] = LATITUDE
            df["longitude"] = LONGITUDE
            # ------------------------------------------------
            # Convert time column
            # ------------------------------------------------
            df["time"] = pd.to_datetime(df["time"])
            # ------------------------------------------------
            # Sort by time
            # ------------------------------------------------
            df = df.sort_values("time").reset_index(drop=True)
            # ------------------------------------------------
            # Remove duplicate timestamps
            # ------------------------------------------------
            df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
            # ------------------------------------------------
            # Save CSV
            # ------------------------------------------------
            df.to_csv(OUTPUT_FILE,index=False)
            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------
            print("\n[SUCCESS]")
            print(
                f"Rows       : {len(df):,}"
            )
            print(
                f"Columns    : {len(df.columns)}"
            )
            print(
                f"Start      : {df['time'].min()}"
            )
            print(
                f"End        : {df['time'].max()}"
            )
            print(
                f"File       : {OUTPUT_FILE}"
            )
            print("\nColumns:")
            for column in df.columns:
                print(
                    f"  - {column}"
                )
            print("=" * 70)
            return True
        except requests.exceptions.RequestException as error:
            print(f"[ERROR] Request failed:")
            print(error)
            if attempt < max_retries:
                wait_time = 30 * attempt
                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                print("[FAILED] Maximum retries reached.")
                return False
        except Exception as error:
            print("[ERROR] Unexpected error:")
            print(error)
            return False
    return False
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    success = download_hanoi()
    if success:
        print("\n[DONE] Hanoi dataset downloaded successfully.")
    else:
        print("\n[FAILED] Hanoi dataset download failed.")