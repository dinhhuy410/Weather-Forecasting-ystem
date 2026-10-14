"""Shared schema and time checks for collection, processing and forecasting."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    "weather_code", "is_day",
]
METADATA_COLUMNS = ["time", "city", "latitude", "longitude"]
TIME_FEATURES = ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
HISTORY_FEATURES = [
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h",
    "temp_delta_1h", "temp_rolling_mean_3h",
]
WIND_DIRECTION_COLUMNS = ["wind_direction_10m", "wind_direction_100m"]
FEATURE_ENGINEERING_VERSION = 1


def engineer_forecast_features(frame: pd.DataFrame, history=None) -> pd.DataFrame:
    """Create features available at observation time, using only current/past data.

    Without history, discard exactly the first 24 warm-up hours. For external
    data, supply the preceding raw observations so the first test hour is kept.
    Calendar components use Hanoi local time; yearly phase handles leap years.
    Raw wind angles remain for auditing; model candidates use their sin/cos.
    """
    validate_hourly_frame(frame, ["temperature_2m"])
    working = frame.copy().reset_index(drop=True)
    working["time"] = pd.to_datetime(working["time"])
    context_length = 0
    if history is not None:
        validate_hourly_frame(history, ["temperature_2m"])
        context = history[["time", "temperature_2m"]].tail(24).copy()
        context["time"] = pd.to_datetime(context["time"])
        if len(context) < 24 or context["time"].iloc[-1] + pd.Timedelta(hours=1) != working["time"].iloc[0]:
            raise ValueError("Feature history must contain 24 hours immediately preceding the input.")
        context_length = len(context)
    times = working["time"]
    local_times = times.dt.tz_convert("Asia/Ho_Chi_Minh") if times.dt.tz is not None else times
    hour_phase = 2 * np.pi * local_times.dt.hour / 24
    days_in_year = np.where(local_times.dt.is_leap_year, 366, 365)
    year_phase = 2 * np.pi * (local_times.dt.dayofyear - 1 + local_times.dt.hour / 24) / days_in_year
    working["hour_sin"], working["hour_cos"] = np.sin(hour_phase), np.cos(hour_phase)
    working["day_of_year_sin"], working["day_of_year_cos"] = np.sin(year_phase), np.cos(year_phase)
    temperature = working["temperature_2m"].reset_index(drop=True)
    if history is not None:
        temperature = pd.concat([context["temperature_2m"], temperature], ignore_index=True)
    for lag in (1, 3, 24):
        working[f"temp_lag_{lag}h"] = temperature.shift(lag).iloc[context_length:].to_numpy()
    working["temp_delta_1h"] = temperature.diff().iloc[context_length:].to_numpy()
    working["temp_rolling_mean_3h"] = temperature.rolling(3, min_periods=3).mean().iloc[context_length:].to_numpy()
    for column in WIND_DIRECTION_COLUMNS:
        if column in working:
            phase = np.deg2rad(working[column])
            working[f"{column}_sin"], working[f"{column}_cos"] = np.sin(phase), np.cos(phase)
    if history is None:
        working = working.iloc[24:].reset_index(drop=True)
    validate_hourly_frame(working, [*TIME_FEATURES, *HISTORY_FEATURES])
    return working


def validate_hourly_frame(frame: pd.DataFrame, required_columns=()) -> None:
    """Reject missing/duplicate hours instead of treating the next row as +1h."""
    missing = set(["time", *required_columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError("At least two hourly rows are required.")
    times = pd.to_datetime(frame["time"], errors="coerce")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Invalid or duplicate timestamps.")
    if not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Data must be sorted and continuous at hourly intervals.")
    if frame.isna().any().any():
        raise ValueError("Missing values remain; do not impute future labels.")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite numeric values in dataset.")


def load_feature_selection(path: Path, target: str, horizon: int, dataset_path=None, *, model_family="ml") -> list[str]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report["target"] != target or report["horizon_hours"] != horizon:
        raise ValueError("Target/horizon differs from the feature selection run.")
    if model_family not in ("ml", "dl"):
        raise ValueError("Model family must be ml or dl.")
    features = report.get("dl_selected_features", report["selected_features"]) if model_family == "dl" else report["selected_features"]
    if not features or len(features) != len(set(features)) or target not in features:
        raise ValueError("Selected features must be unique and include the current target.")
    if set(features) & set(METADATA_COLUMNS):
        raise ValueError("Metadata must not be used as model features.")
    if dataset_path is not None:
        digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
        if report.get("processed_sha256") != digest:
            raise ValueError("Dataset differs from its feature selection report; rerun clean_raw.")
    return features


def build_forecast_dataset(frame, features, target, horizon, sequence_length=1):
    """Flatten the past window ending at each input time; label horizon hours later.

    Current-hour columns keep their names. Older values use ``<feature>_lag_<n>h``.
    A 24-hour window contains the current hour and lags 1 through 23, with stride 1.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer.")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("Sequence length must be a positive integer.")
    validate_hourly_frame(frame, features)
    if len(frame) < sequence_length + horizon:
        raise ValueError("Not enough rows for the requested window and forecast horizon.")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in features):
        raise ValueError("All model features must be numeric.")
    label = f"{target}_future_{horizon}h"
    parts = [frame[features]]
    for lag in range(1, sequence_length):
        parts.append(frame[features].shift(lag).rename(
            columns={feature: f"{feature}_lag_{lag}h" for feature in features},
        ))
    data = pd.concat(parts, axis=1)
    feature_columns = data.columns.tolist()
    data["time"] = pd.to_datetime(frame["time"])
    data["label_time"] = data["time"].shift(-horizon)
    data[label] = frame[target].shift(-horizon)
    data = data.iloc[sequence_length - 1:-horizon].reset_index(drop=True)
    return data[["time", "label_time", *feature_columns, label]], label
