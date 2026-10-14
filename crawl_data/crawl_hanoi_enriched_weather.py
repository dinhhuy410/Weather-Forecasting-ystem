from pathlib import Path
import sys
import time

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.weather_data import WEATHER_COLUMNS, validate_hourly_frame

CITY_NAME = "Hanoi"
LATITUDE = 21.0285
LONGITUDE = 105.8542
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/Hanoi.csv"
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_archive(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_COLUMNS),
        "timezone": "Asia/Ho_Chi_Minh", "models": "era5",
        "temperature_unit": "celsius", "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    for attempt in range(3):
        try:
            response = requests.get(API_URL, params=params, timeout=300)
            response.raise_for_status()
            payload = response.json()
            if "hourly" not in payload:
                raise ValueError("Archive API response is missing hourly data.")
            frame = pd.DataFrame(payload["hourly"])
            frame["time"] = pd.to_datetime(frame["time"], errors="raise")
            frame["city"] = CITY_NAME
            frame["latitude"] = LATITUDE
            frame["longitude"] = LONGITUDE
            validate_hourly_frame(frame, WEATHER_COLUMNS)
            expected = pd.date_range(start_date, pd.Timestamp(end_date) + pd.Timedelta(hours=23), freq="h")
            if not pd.DatetimeIndex(frame["time"]).equals(expected):
                raise ValueError("Archive response does not cover the complete requested period.")
            return frame[["time", *WEATHER_COLUMNS, "city", "latitude", "longitude"]]
        except requests.exceptions.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if attempt == 2 or (status is not None and status != 429 and status < 500):
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Archive download failed.")


def download_archive(start_date: str, end_date: str, output_file: Path) -> bool:
    output_file = Path(output_file)
    temporary = output_file.with_suffix(".csv.tmp")
    try:
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if start > end:
            raise ValueError("start_date must be before end_date.")
        frames = []
        for year in range(start.year, end.year + 1):
            left = max(start, pd.Timestamp(year=year, month=1, day=1))
            right = min(end, pd.Timestamp(year=year, month=12, day=31))
            print(f"Download ERA5 {left.date()} -> {right.date()}", flush=True)
            frames.append(fetch_archive(str(left.date()), str(right.date())))
        frame = pd.concat(frames, ignore_index=True)
        validate_hourly_frame(frame, WEATHER_COLUMNS)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(temporary, index=False)
        temporary.replace(output_file)
        print(f"Saved {len(frame):,} rows x {len(frame.columns)} columns: {output_file}")
        return True
    except (requests.exceptions.RequestException, ValueError, KeyError, OSError) as error:
        print(f"Download failed; existing dataset preserved: {error}", file=sys.stderr)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def crawl_hanoi_enriched_weather() -> bool:
    return download_archive(START_DATE, END_DATE, OUTPUT_FILE)


if __name__ == "__main__":
    raise SystemExit(0 if crawl_hanoi_enriched_weather() else 1)
