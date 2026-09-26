import pandas as pd
import glob
import os


files = glob.glob("data/raw/*.csv")

print("=" * 70)
print("WEATHER DATASET INSPECTION")
print("=" * 70)

print(f"\nNumber of files: {len(files)}")

total_rows = 0

for file in files:

    df = pd.read_csv(file)

    rows = len(df)
    total_rows += rows

    print("\n" + "-" * 70)
    print(f"File: {os.path.basename(file)}")
    print(f"Rows: {rows:,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTime range:")
    print(df["time"].min())
    print(df["time"].max())


print("\n" + "=" * 70)
print(f"TOTAL ROWS: {total_rows:,}")
print("=" * 70)