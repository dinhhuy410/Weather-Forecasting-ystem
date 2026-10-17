# ML bằng cửa sổ trượt theo năm

Mở [ml_models.ipynb](../train_model/ml_models.ipynb) và chạy từ trên xuống. Đây là file duy nhất trong `train_model`; toàn bộ luồng xử lý và các lời gọi `RandomForestRegressor`, `XGBRegressor`, `.fit()`, `.predict()` nằm trong notebook.

Cell đầu chỉ import thư viện. Các cell sau lần lượt cấu hình → đọc dữ liệu → chọn feature bằng train → tạo cửa sổ → chia tập → baseline → khởi tạo/train RF → khởi tạo/train XGBoost → chọn/lưu model → đọc test → dự đoán/đánh giá → biểu đồ. Sửa tham số model ngay trước cell train tương ứng.

Cần các thư viện trong `requirements.txt`, môi trường Python 3.12. Có thể thực thi notebook từ thư mục gốc bằng:

```bash
jupyter nbconvert --to notebook --execute train_model/ml_models.ipynb --inplace
```

## Chia tập và chống rò rỉ

`temperature_2m` là mục tiêu, đơn vị °C. Với giờ dự báo t, input gồm các biến thời tiết từ t−23 đến t, nhãn là nhiệt độ t+1. Bước trượt bằng 1. Input được trải phẳng để dùng với Random Forest và XGBoost.

| Tập | Giờ của nhãn | Cách dùng |
|---|---|---|
| Train | Trước 01/01/2025 | Fit các model; xác định cột hằng |
| Validation | Năm 2025 | Chọn MAE nhỏ nhất, hòa thì RMSE nhỏ nhất |
| Test | Từ 01/01/2026 đến hết dữ liệu 2026 trong CSV | Đánh giá model đã chốt và baseline persistence |

Model không được fit lại trên 2025. Trong notebook, test chỉ được mở ở mục 12, sau khi mục 11 đã lưu model và `selection.json`. Các cửa sổ đầu validation/test giữ lịch sử cuối tập trước. Các giờ test đã quan sát được có thể vào cửa sổ của lần dự báo tiếp theo; không cập nhật model. Đây là đánh giá dự báo từng giờ, không phải dự báo cả năm từ ngày 01/01.

Nếu đổi horizon H>1, code loại các nhãn train/validation sát biên chưa có tại thời điểm dự báo đầu tập sau. Mọi nhãn dùng fit/chọn model phải có thời điểm ≤ mốc dự báo đầu tập sau. Các cửa sổ vẫn có thể chia sẻ quan sát quá khứ.

Danh sách biến thời tiết hiện tại được định nghĩa trong code, không đọc `data_processing/feature_selection.json`. Calendar được tạo lại; không dùng cột lag/rolling hoặc nhãn tương lai trong CSV cũ. Cột hằng chỉ được xác định bằng dữ liệu đã có trước validation. Không scale. Chỉ forward-fill predictor tối đa 3 giờ; không điền nhiệt độ, backward-fill hay nội suy từ tương lai. Giờ thiếu/trùng hoặc giá trị không hữu hạn gây lỗi.

Input mặc định là `data/processed/Hanoi_final.csv` và `data/processed/Hanoi_test_final.csv`. Processed bắt đầu từ 02/01/2010 vì bước tiền xử lý cũ bỏ 24 giờ đầu. Pipeline tiếp tục bỏ số giờ khởi động cần cho cửa sổ. Raw hiện là con trỏ Git LFS. Khi đã có CSV raw thật, đổi `DEVELOPMENT_PATH` và `TEST_PATH` ở cell cấu hình sang hai đường dẫn raw.

## Chạy riêng hai bước

Chạy đến hết mục 11 để train, chọn và lưu model mà chưa đọc test. Chạy tiếp mục 12–14 để mở test 2026, dự đoán và vẽ biểu đồ. Chạy các cell theo thứ tự trong cùng kernel.

Kết quả nằm tại `artifacts/ml_sliding_window/` ở thư mục gốc dự án. Hướng dẫn nằm trong `docs/`; `train_model/` chỉ chứa notebook.

Mỗi lần train tạo thư mục mới, giữ nguyên các kết quả trước. Các file chính:

| File | Nội dung |
|---|---|
| `best_model.joblib` | Model đã fit trên <=2024, cấu hình và schema input |
| `selection.json` | Lựa chọn bằng 2025, hash dữ liệu/model và phiên bản thư viện |
| `history_context.csv` | Lịch sử cuối 2025 để dự báo từ giờ đầu 2026 |
| `data_splits.csv` | Số mẫu và biên thời gian train/validation |
| `validation_metrics.csv`, `validation_predictions.csv` | So sánh các cấu hình trên 2025 |
| `test_metrics.csv`, `test_metrics_by_month.csv` | MAE, RMSE và R² test |
| `test_predictions.csv`, `evaluation.json` | Dự đoán từng giờ, biên test và hash test |
| `test_2026.png` | Đồ thị do notebook tạo |

Baseline persistence dự báo nhiệt độ tương lai bằng nhiệt độ hiện tại. MAE và RMSE tính theo °C. Không dùng điểm test để chọn lại cấu hình sau khi đọc kết quả.

## Kết quả lượt chạy 17/10/2026

Notebook đã chạy đầy đủ với cửa sổ 24 giờ, horizon 1 giờ và seed 42. Có 131.448 mẫu train, 8.760 mẫu validation và 6.528 mẫu test. Nhãn test từ 01/01/2026 00:00 đến 29/09/2026 23:00 theo giờ Hà Nội.

| Cấu hình | MAE validation 2025 | RMSE validation 2025 | Được chọn |
|---|---:|---:|---|
| RF depth 12 | 0,4349 | 0,5925 | Không |
| RF depth 20 | 0,4016 | 0,5507 | Không |
| XGBoost depth 4 | 0,2974 | 0,4214 | Không |
| XGBoost depth 6 | 0,2774 | 0,3972 | Có |

| Model đánh giá trên test 2026 | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| XGBoost depth 6 đã chốt | 0,3749 | 0,5867 | 0,9880 |
| Persistence | 0,5841 | 0,9396 | 0,9692 |

Model giảm MAE khoảng 35,8% so với persistence trên cùng giờ test. Toàn bộ model, dự đoán và biểu đồ của lượt notebook gọi thư viện trực tiếp nằm trong [thư mục kết quả](../artifacts/ml_sliding_window/20261017_223841_notebook_5p5kcsof/). Cả 20 code cell đã chạy thành công; notebook có sẵn bảng và biểu đồ.

```bash
python3.12 -m unittest discover -s tests -v
```

Kiểm tra gồm giá trị cửa sổ, tương lai không thay đổi input quá khứ, split theo năm nhãn, horizon dài, cột hằng chỉ dựa trên train, causal fill, năm nhuận, lịch sử qua đầu năm và pipeline nhỏ chạy cả RF/XGBoost. Kiểm tra pipeline xác nhận chỉ fit trên train, chưa đọc test khi chọn model và không refit sau đó.

Test 2026 đã được xem ở các thử nghiệm cũ; không thể xem đây là tập hoàn toàn chưa từng được sử dụng. Dữ liệu ERA5 là tái phân tích: chất lượng trên backtest này không tự chứng minh chất lượng realtime.

Tham khảo: [scikit-learn về data leakage](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage), [feature trễ cho dự báo chuỗi thời gian](https://scikit-learn.org/1.7/auto_examples/applications/plot_time_series_lagged_features.html).
