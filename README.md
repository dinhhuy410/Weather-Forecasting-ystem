# Weather-Forecasting-ystem

Dự án xây dựng quy trình Data Science cho bài toán dự báo thời tiết Hà Nội từ dữ liệu hourly. Dữ liệu được thu thập từ Open-Meteo, xử lý sạch, phân tích EDA, sau đó huấn luyện nhóm mô hình Machine Learning và Deep Learning.

## Mục Tiêu

- Thu thập dữ liệu thời tiết Hà Nội theo giờ.
- Kiểm tra chất lượng dữ liệu: missing values, duplicate timestamps, khoảng thời gian và thống kê mô tả.
- Tạo đặc trưng thời gian, lag features và rolling statistics cho bài toán dự báo chuỗi thời gian.
- Huấn luyện và so sánh Random Forest với XGBoost.
- Huấn luyện và so sánh LSTM với GRU trên dữ liệu dạng chuỗi.
- Đánh giá mô hình bằng MAE, Median AE, RMSE, R2 và MAPE.
- Bổ sung TimeSeriesSplit cross-validation, hyperparameter tuning, feature importance và error analysis.

## Cấu Trúc Dự Án

```text
Weather-Forecasting-ystem-main/
├── README.md
├── requirements.txt
├── crawl_data/
│   ├── download_hanoi.py
│   └── README.md
├── src/
│   ├── download.py
│   ├── inspect_data.py
│   └── preprocess.py
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── Hanoi.csv
│   └── processed/
│       ├── Hanoi_clean.csv
│       └── Hanoi_final.csv
├── data_processing/
│   ├── clean_raw.ipynb
│   ├── prepare_final.ipynb
│   └── README.md
├── eda/
│   ├── data_analyze.ipynb
│   ├── eda_visualization.ipynb
│   └── README.md
└── train_model/
    ├── ml_models.ipynb
    ├── dl_models.ipynb
    └── README.md
```

## Dữ Liệu

Dataset raw nằm tại `data/raw/Hanoi.csv`. Dataset sau xử lý được tạo ra trong `data/processed/`.

| Thông tin | Giá trị |
|---|---:|
| Khu vực | Hà Nội |
| Tần suất | Theo giờ |
| Target mặc định | `temperature_2m` |
| Forecast horizon mặc định | 1 giờ tiếp theo |
| Số dòng hiện tại | 140,256 |

## Quy Trình Xử Lý

1. **Data Collection**: tải dữ liệu bằng `crawl_data/download_hanoi.py`.
2. **Data Cleaning**: chạy `data_processing/clean_raw.ipynb` để tạo `Hanoi_clean.csv`.
3. **Prepare Final Dataset**: chạy `data_processing/prepare_final.ipynb` để tạo `Hanoi_final.csv`.
4. **EDA Analysis**: chạy `eda/data_analyze.ipynb`.
5. **EDA Visualization**: chạy `eda/eda_visualization.ipynb`.
6. **Machine Learning Modeling**: chạy `train_model/ml_models.ipynb`.
7. **Deep Learning Modeling**: chạy `train_model/dl_models.ipynb`.

## Cách Chạy Project

### 1. Cài môi trường Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Crawl dữ liệu

```bash
python3 crawl_data/download_hanoi.py
```

Nếu đã có sẵn `data/raw/Hanoi.csv` thì có thể bỏ qua bước crawl.

### 3. Kiểm tra dữ liệu raw

```bash
python3 src/inspect_data.py
```

### 4. Chạy notebook theo thứ tự

```text
data_processing/clean_raw.ipynb
data_processing/prepare_final.ipynb
eda/data_analyze.ipynb
eda/eda_visualization.ipynb
train_model/ml_models.ipynb
train_model/dl_models.ipynb
```

Notebook `ml_models.ipynb` dùng cho Random Forest và XGBoost. Notebook `dl_models.ipynb` dùng cho LSTM và GRU.
