import pandas as pd
import os
# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = "data/raw/Hanoi.csv"
# ============================================================
# HELPER FUNCTIONS
# ============================================================
def print_separator():
    print("\n" + "=" * 70)
def inspect_dataset(file_path):
    """
    Inspect the Hanoi weather dataset.
    """
    print_separator()
    print("HANOI WEATHER DATASET INSPECTION")
    print_separator()
    # -------------------------------------------------------
    # Check file
    # --------------------------------------------------------
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False
    print(f"\nFile: {file_path}")
    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        print(f"[ERROR] Cannot read CSV file: {error}")
        return False
    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. BASIC INFORMATION")
    print("-" * 70)
    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Memory     : {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. COLUMNS")
    print("-" * 70)
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")
    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. DATA TYPES")
    print("-" * 70)
    print(df.dtypes.to_string())
    # --------------------------------------------------------
    # Time information
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. TIME INFORMATION")
    print("-" * 70)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce"
        )
        invalid_time = df["time"].isna().sum()
        print(f"Start time       : {df['time'].min()}")
        print(f"End time         : {df['time'].max()}")
        print(f"Invalid timestamps: {invalid_time:,}")
        # ----------------------------------------------------
        # Check duplicated timestamps
        # ----------------------------------------------------
        duplicate_time = df["time"].duplicated().sum()
        print(
            f"Duplicate timestamps: "
            f"{duplicate_time:,}"
        )
        # ----------------------------------------------------
        # Check time intervals
        # ----------------------------------------------------
        sorted_time = (
            df["time"]
            .dropna()
            .sort_values()
        )
        if len(sorted_time) > 1:
            time_difference = (
                sorted_time
                .diff()
                .dropna()
            )
            print(
                f"Most common interval: "
                f"{time_difference.mode().iloc[0]}"
            )
            unexpected_intervals = (
                time_difference
                != pd.Timedelta(hours=1)
            ).sum()
            print(
                f"Non-hourly intervals: "
                f"{unexpected_intervals:,}"
            )
    else:
        print("[WARNING] 'time' column not found.")
    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. MISSING VALUES")
    print("-" * 70)
    missing_count = df.isnull().sum()
    missing_percent = (
        missing_count
        / len(df)
        * 100
    )
    missing_table = pd.DataFrame(
        {
            "Missing": missing_count,
            "Missing_%": missing_percent.round(4),
        }
    )
    missing_table = missing_table.sort_values(
        by="Missing",
        ascending=False
    )
    print(missing_table.to_string())
    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. DUPLICATE ROWS")
    print("-" * 70)
    duplicate_rows = df.duplicated().sum()
    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )
    # --------------------------------------------------------
    # Constant columns
    # -------------------------------------------------------
    print("\n" + "-" * 70)
    print("7. CONSTANT COLUMNS")
    print("-" * 70)
    constant_columns = []
    for column in df.columns:
        unique_count = df[column].nunique(
            dropna=False
        )
        if unique_count <= 1:
            constant_columns.append(column)
    if constant_columns:
        print(
            "Columns with only one unique value:"
        )
        for column in constant_columns:
            print(f"  - {column}")
    else:
        print("No constant columns found.")
    # --------------------------------------------------------
    # Numerical statistics
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("8. NUMERICAL STATISTICS")
    print("-" * 70)
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns
    if len(numeric_columns) > 0:
        statistics = df[numeric_columns].describe().T
        statistics = statistics[
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
        print(
            statistics.round(3).to_string()
        )
    else:
        print("No numerical columns found.")
    # --------------------------------------------------------
    # City information
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("9. LOCATION INFORMATION")
    print("-" * 70)
    if "city" in df.columns:
        print(
            "Cities:"
        )
        print(
            df["city"]
            .value_counts(dropna=False)
            .to_string()
        )
    if "latitude" in df.columns:
        print(
            f"Latitude: "
            f"{df['latitude'].dropna().unique()}"
        )
    if "longitude" in df.columns:
        print(
            f"Longitude: "
            f"{df['longitude'].dropna().unique()}"
        )
    # --------------------------------------------------------
    # Weather feature groups
    # -------------------------------------------------------
    print("\n" + "-" * 70)
    print("10. FEATURE GROUPS")
    print("-" * 70)
    feature_groups = {
        "Temperature": [
            "temperature_2m",
            "dew_point_2m",
            "apparent_temperature",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm",
        ],
        "Humidity": [
            "relative_humidity_2m",
            "vapour_pressure_deficit",
        ],
        "Precipitation": [
            "precipitation",
            "rain",
            "snowfall",
        ],
        "Pressure": [
            "pressure_msl",
            "surface_pressure",
        ],
        "Cloud": [
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
        ],
        "Wind": [
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_10m",
            "wind_direction_100m",
            "wind_gusts_10m",
        ],
        "Soil moisture": [
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm",
        ],
        "Other": [
            "et0_fao_evapotranspiration",
            "weather_code",
            "is_day",
        ],
    }
    for group_name, columns in feature_groups.items():
        available_columns = []
        for column in columns:
            if column in df.columns:
                available_columns.append(column)
        print(
            f"\n{group_name}: "
            f"{len(available_columns)} features"
        )
        for column in available_columns:
            print(f"  - {column}")
    # --------------------------------------------------------
    # Dataset quality summary
    # --------------------------------------------------------
    print_separator()
    print("DATASET QUALITY SUMMARY")
    print_separator()
    total_missing = df.isnull().sum().sum()
    print(
        f"Total rows           : {len(df):,}"
    )
    print(
        f"Total columns        : {len(df.columns)}"
    )
    print(
        f"Total missing values : {total_missing:,}"
    )
    print(
        f"Duplicate rows       : {duplicate_rows:,}"
    )
    if "time" in df.columns:
        print(
            f"Duplicate timestamps : "
            f"{df['time'].duplicated().sum():,}"
        )
    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------
    has_missing = total_missing > 0
    has_duplicates = duplicate_rows > 0
    print("\nSTATUS:")
    if not has_missing and not has_duplicates:
        print(
            "[OK] Dataset has no missing values "
            "and no duplicate rows."
        )
    elif has_missing:
        print(
            "[WARNING] Dataset contains missing values."
        )
    elif has_duplicates:
        print(
            "[WARNING] Dataset contains duplicate rows."
        )
    print_separator()
    return True
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    inspect_dataset(DATA_FILE)