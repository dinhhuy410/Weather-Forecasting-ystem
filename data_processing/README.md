# Data Processing

Thư mục này chứa hai notebook xử lý dữ liệu theo thứ tự pipeline rõ ràng.

```text
data_processing/
├── clean_raw.ipynb
├── prepare_final.ipynb
└── README.md
```

## Clean Raw

`clean_raw.ipynb`

- Input: `../data/raw/Hanoi.csv`
- Output: `../data/processed/Hanoi_clean.csv`
- Vai trò: parse thời gian, sắp xếp dữ liệu, xoá timestamp trùng, chuẩn hoá kiểu số và xử lý missing values.

## Prepare Final

`prepare_final.ipynb`

- Input: `../data/processed/Hanoi_clean.csv`
- Output: `../data/processed/Hanoi_final.csv`
- Vai trò: thêm biến lịch cơ bản, kiểm tra lại schema, bỏ cột hằng nếu không hữu ích và tạo dataset cuối cho EDA/modeling.
