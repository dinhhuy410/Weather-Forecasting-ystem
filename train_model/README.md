# Model Training

Thư mục này chứa notebook huấn luyện mô hình dự báo thời tiết Hà Nội, tách riêng nhóm Machine Learning và Deep Learning.

```text
train_model/
├── ml_models.ipynb
├── dl_models.ipynb
└── README.md
```

## Dữ Liệu Và Target

- Input chính: `../data/processed/Hanoi_final.csv`
- Target mặc định: `temperature_2m`
- Forecast horizon mặc định: `1` giờ tiếp theo
- Train/test split: 80/20 theo thứ tự thời gian
- ML features: biến thời tiết hiện tại, time features, lag features và rolling statistics
- DL features: chuỗi nhiều timestep để train LSTM và GRU

## File Chính

| Notebook | Vai trò |
|---|---|
| `ml_models.ipynb` | Train và so sánh Random Forest, XGBoost bằng TimeSeriesSplit, tuning, feature importance và error analysis. |
| `dl_models.ipynb` | Train và so sánh LSTM, GRU trên dữ liệu dạng sequence. |

## Chỉ Số Đánh Giá

- `MAE`: sai số tuyệt đối trung bình.
- `Median AE`: sai số tuyệt đối trung vị, giảm ảnh hưởng của outlier.
- `RMSE`: phạt mạnh các lỗi dự báo lớn.
- `R2`: mức độ giải thích phương sai của mô hình.
- `MAPE (%)`: sai số phần trăm trung bình.

Với MAE, Median AE, RMSE và MAPE thì giá trị càng thấp càng tốt. Với R2 thì giá trị càng cao càng tốt.

## Thứ Tự Chạy Gợi Ý

1. Chạy notebook trong `data_processing/` để tạo `data/processed/Hanoi_final.csv`.
2. Chạy `ml_models.ipynb` để train Random Forest và XGBoost.
3. Chạy `dl_models.ipynb` để train LSTM và GRU.

`ml_models.ipynb` dùng `RandomizedSearchCV`; `dl_models.ipynb` dùng neural network nên cả hai file có thể chạy lâu hơn notebook EDA.
