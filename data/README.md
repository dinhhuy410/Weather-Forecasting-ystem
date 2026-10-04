# Data

Thư mục dữ liệu của project.

```text
data/
├── raw/
│   └── Hanoi.csv
└── processed/
    ├── Hanoi_clean.csv
    └── Hanoi_final.csv
```

## Raw

`data/raw/Hanoi.csv` là dữ liệu hourly tải từ Open-Meteo.

## Processed

Các file processed được tạo bằng notebook trong `data_processing/`:

- `Hanoi_clean.csv`: dữ liệu đã parse thời gian, bỏ trùng, chuẩn hoá kiểu dữ liệu và xử lý missing values.
- `Hanoi_final.csv`: dataset cuối cùng cho EDA và modeling, có thêm biến lịch/cyclical features.
