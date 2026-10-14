from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from crawl_data.crawl_hanoi_enriched_weather import download_archive

START_DATE = "2026-01-01"
END_DATE = "2026-09-29"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/test/Hanoi_test_set.csv"


def crawl_hanoi_test_set() -> bool:
    # Never fill evaluation labels from a forecast model.
    return download_archive(START_DATE, END_DATE, OUTPUT_FILE)


if __name__ == "__main__":
    raise SystemExit(0 if crawl_hanoi_test_set() else 1)
