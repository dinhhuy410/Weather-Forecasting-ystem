# Data Collection

Thư mục này chứa script tải dữ liệu thời tiết Hà Nội từ Open-Meteo Historical Weather API.

```text
crawl_data/
├── download_hanoi.py
└── README.md
```

## Output

- `../data/raw/Hanoi.csv`

## Cách chạy

```bash
python3 crawl_data/download_hanoi.py
```

Script tải dữ liệu hourly từ `2010-01-01` đến `2025-12-31`, thêm thông tin thành phố/toạ độ, sắp xếp theo thời gian và bỏ timestamp trùng.
