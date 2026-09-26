import requests
import pandas as pd
import time
import os


CITIES = {
    "Hanoi": (21.0285, 105.8542),
    "HaiPhong": (20.8449, 106.6881),
    "ThanhHoa": (19.8067, 105.7852),
    "Vinh": (18.6796, 105.6813),
    "Hue": (16.4637, 107.5909),
    "DaNang": (16.0544, 108.2022),
    "NhaTrang": (12.2388, 109.1967),
    "DaLat": (11.9404, 108.4583),
    "HoChiMinh": (10.8231, 106.6297),
    "CanTho": (10.0452, 105.7469),
}


START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "weather_code",
]


OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_city(city_name, latitude, longitude):

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{city_name}.csv"
    )

    # Nếu đã tải rồi thì bỏ qua
    if os.path.exists(output_path):

        try:
            old_df = pd.read_csv(
                output_path,
                nrows=1
            )

            print(
                f"[SKIP] {city_name} already exists"
            )

            return True

        except Exception:
            print(
                f"[WARNING] {city_name} file exists but cannot be read."
            )

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "timezone": "Asia/Bangkok",
        "models": "era5",
    }

    max_retries = 5

    for attempt in range(1, max_retries + 1):

        print(
            f"[{city_name}] "
            f"Attempt {attempt}/{max_retries}"
        )

        try:

            response = requests.get(
                url,
                params=params,
                timeout=180
            )

            # Rate limit
            if response.status_code == 429:

                wait_time = 30 * attempt

                print(
                    f"[429] Rate limit. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            response.raise_for_status()

            data = response.json()

            if "hourly" not in data:

                print(
                    f"[ERROR] No hourly data for {city_name}"
                )

                return False

            df = pd.DataFrame(
                data["hourly"]
            )

            df["city"] = city_name
            df["latitude"] = latitude
            df["longitude"] = longitude

            df.to_csv(
                output_path,
                index=False
            )

            print(
                f"[SUCCESS] {city_name}: "
                f"{len(df):,} rows saved"
            )

            return True

        except requests.exceptions.RequestException as e:

            print(
                f"[ERROR] {city_name}: {e}"
            )

            if attempt < max_retries:

                wait_time = 30 * attempt

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    f"[FAILED] {city_name}"
                )

                return False

    return False


print("=" * 60)
print("VIETNAM WEATHER DATA DOWNLOADER")
print("=" * 60)


for city_name, coordinates in CITIES.items():

    latitude, longitude = coordinates

    success = download_city(
        city_name,
        latitude,
        longitude
    )

    # Nghỉ giữa các thành phố
    print(
        "\nWaiting 20 seconds before next city...\n"
    )

    time.sleep(20)


print("=" * 60)
print("DOWNLOAD COMPLETED")
print("=" * 60)