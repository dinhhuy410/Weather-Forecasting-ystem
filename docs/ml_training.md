# ML bằng cửa sổ trượt theo năm

Mở [ml_models.ipynb](../train_model/ml_models.ipynb) và chạy từ trên xuống. Đây là notebook chính trong `train_model`; toàn bộ luồng xử lý và các lời gọi `RandomForestRegressor`, `XGBRegressor`, `.fit()`, `.predict()` nằm trong notebook.

Cell đầu chỉ import thư viện. Các cell sau lần lượt cấu hình → đọc dữ liệu → khai báo feature → tạo cửa sổ và chia tập → baseline → khởi tạo/train RF → khởi tạo/train XGBoost → chọn model → feature importance → đọc test → dự đoán/đánh giá → biểu đồ. Sửa tham số model ngay trước cell train tương ứng.

Cần các thư viện trong `requirements.txt`, môi trường Python 3.12. Có thể thực thi notebook từ thư mục gốc bằng:

```bash
jupyter nbconvert --to notebook --execute train_model/ml_models.ipynb --inplace
```

## Chia tập và chống rò rỉ

`temperature_2m` là mục tiêu, đơn vị °C. Với giờ dự báo t, input gồm các biến thời tiết từ t−23 đến t, nhãn là nhiệt độ t+1. Bước trượt bằng 1. Các cell tạo dữ liệu phát triển và test gọi trực tiếp `sliding_window_view()` của NumPy. Phần code còn lại ghép nhãn theo horizon, giữ thông tin thời gian để chia tập và trải phẳng input cho Random Forest/XGBoost.

Mục 5 tách riêng từng bước tạo cửa sổ, đổi chiều, trải phẳng thành `X` và lấy `y = temperature[WINDOW_HOURS:]` cho horizon 1 giờ. Sau đó gọi ngay `train_test_split(..., shuffle=False)` trong cùng cell để có `X_train`, `X_valid`, `y_train`, `y_valid`, theo mốc thời gian của nhãn. Mục tiếp theo tính baseline.

| Tập | Giờ của nhãn | Cách dùng |
|---|---|---|
| Train | Trước 01/01/2025 | Fit các model |
| Validation | Năm 2025 | Chọn riêng một RF và một XGBoost: MAE nhỏ nhất, hòa thì RMSE nhỏ nhất |
| Test | Từ 01/01/2026 đến hết dữ liệu 2026 trong CSV | Đánh giá đúng hai model đã chọn và baseline persistence |

`ParameterGrid` tạo mọi tổ hợp trong các khoảng tham số dưới đây:

| Model | Độ sâu (`max_depth`) | Số cây (`n_estimators`) | Learning rate | Tổng cấu hình |
|---|---|---|---|---|
| RF | 12, 16, 20 | 300 | Không áp dụng | 3 |
| XGBoost | 6, 8, 10 | 600 | 0,03 | 3 |

Grid hiện tại có tổng cộng 6 cấu hình, tập trung quanh các độ sâu đã cho kết quả tốt trong lượt trước. Sau khi đổi grid, chạy lại từ mục 7 để cập nhật bảng điểm, hai model được chọn và các biểu đồ. Các kết quả lịch sử bên dưới thuộc lượt chạy cũ với grid rộng hơn.

Grid gọn đã được chạy mới đầy đủ: cả 6 cấu hình được fit lại trên toàn bộ train, sau đó chọn model, tính importance và đánh giá test. RF tốt nhất là sâu 20/300 cây (MAE validation 0,395357°C; MAE test 0,458063°C). XGBoost tốt nhất là sâu 8/600 cây/learning rate 0,03 (MAE validation 0,267457°C; MAE test 0,368951°C). Toàn bộ 19 code cell đã chạy thành công. Xem [báo cáo và kết quả đầy đủ](../artifacts/ml_sliding_window/run_qhsreslc/run_report.md). Các kết quả RF sâu 16 ở phần lịch sử thuộc lượt chạy cũ.

Mỗi cấu hình fit trên cùng train và chấm trên validation 2025. Bảng validation lưu tham số và điểm số của các cấu hình đã chạy. Mục 7–8 hiển thị điểm theo MAE/RMSE tăng dần; cell chọn model ở mục 9 hiển thị hai model đã chọn, tiếp theo là importance ở mục 9.1. Chỉ model và dự đoán của ứng viên tốt nhất mỗi loại được giữ lại. Đây là lựa chọn tốt nhất trong các cấu hình đã thử.

Hai model không được fit lại trên 2025. Test chỉ được mở ở mục 10, sau khi mục 9 đã chọn một RF và một XGBoost trong `selected_models`. Các cửa sổ đầu validation/test giữ lịch sử cuối tập trước. Các giờ test đã quan sát được có thể vào cửa sổ của lần dự báo tiếp theo; không cập nhật model. Đây là đánh giá dự báo từng giờ, không phải dự báo cả năm từ ngày 01/01.

Luồng chia tập hiện tại dùng `HORIZON_HOURS = 1`, dự báo sau 1 giờ; không có bước cắt lại train/validation sau `train_test_split`.

Danh sách biến thời tiết hiện tại được định nghĩa trong code, không đọc `data_processing/feature_selection.json`. Dùng trực tiếp `INPUT_FEATURES`, gồm các biến thời tiết và sin/cos thời gian có sẵn trong CSV processed; không dùng cột lag/rolling hoặc nhãn tương lai trong CSV cũ. Không scale hoặc điền thiếu trong notebook; bước tạo cửa sổ dùng trực tiếp CSV processed đã làm sạch.

Input mặc định là `data/processed/Hanoi_final.csv` và `data/processed/Hanoi_test_final.csv`. Processed bắt đầu từ 02/01/2010 vì bước tiền xử lý cũ bỏ 24 giờ đầu. Pipeline tiếp tục bỏ số giờ khởi động cần cho cửa sổ. Raw hiện là con trỏ Git LFS. Nếu khôi phục CSV raw thật, cần tạo các feature sin/cos và xuất lại hai CSV processed trước khi chạy notebook; đổi đường dẫn trực tiếp sang raw sẽ thiếu 8 cột sin/cos.

## Chạy riêng hai bước

Chạy đến hết mục 9 để train và chọn model mà chưa đọc test. Chạy tiếp mục 10–12 để mở test 2026, dự đoán và vẽ biểu đồ. Chạy các cell theo thứ tự trong cùng kernel; hai model được dùng trực tiếp từ `selected_models`.

Mục 9.1 vẽ riêng feature importance cho RF và XGBoost đã chọn: RF dùng `feature_importances_` (MDI), XGBoost dùng `total_gain`. Cộng điểm qua 24 vị trí thời gian để thu được điểm cho từng feature gốc, rồi chuẩn hóa thành 100% riêng trong từng model. Biểu đồ hiển thị đủ 32 feature. Các cell này dùng model đã fit, không đọc test hoặc train lại. Hai loại điểm có định nghĩa khác nhau; không so trực tiếp tỷ lệ importance giữa RF và XGBoost.

Mục 10 ghép `data.tail(WINDOW_HOURS)` vào trước test để tạo `X_test`, `y_test`. Mục 11 chỉ test một RF và một XGBoost đã chọn từ validation, thêm persistence để đối chiếu, rồi lưu CSV. Thống kê theo tháng và biểu đồ so sánh cả hai model được tính ở mục 12.

Kết quả nằm tại `artifacts/ml_sliding_window/` ở thư mục gốc dự án. Hướng dẫn nằm trong `docs/`; `train_model/` chỉ chứa notebook.

Mỗi lần train tạo thư mục mới, giữ nguyên các kết quả trước. Các file chính:

| File | Nội dung |
|---|---|
| `validation_metrics.csv`, `validation_predictions.csv` | So sánh các cấu hình trên 2025 |
| `rf_feature_importance.csv`, `rf_feature_importance.png` | Bảng và biểu đồ importance của RF đã chọn, gộp theo feature gốc |
| `xgb_feature_importance.csv`, `xgb_feature_importance.png` | Bảng và biểu đồ total gain của XGBoost đã chọn, gộp theo feature gốc |
| `test_metrics.csv`, `test_metrics_by_month.csv` | MAE, RMSE và R² test |
| `test_predictions.csv` | Mốc thời gian, nhiệt độ thực tế, dự báo RF, XGBoost và persistence |
| `test_2026.png` | Đồ thị do notebook tạo |

Baseline persistence dự báo nhiệt độ tương lai bằng nhiệt độ hiện tại. MAE và RMSE tính theo °C. Không dùng điểm test để chọn lại cấu hình sau khi đọc kết quả.

## Kết quả chốt ngày 22/10/2026

Theo yêu cầu, dừng tìm kiếm sau **82 cấu hình hoàn tất: 49 RF và 33 XGBoost**, trong dải dự kiến 219 cấu hình. Chọn riêng một model mỗi loại bằng MAE validation, hòa thì xét RMSE; sau đó đánh giá đúng hai model đã chọn trên 6.528 mẫu test 2026. Model vẫn chỉ fit trên 131.448 mẫu train trước 2025; validation có 8.760 mẫu.

| Model | Cấu hình | MAE validation (°C) | MAE test (°C) | RMSE test (°C) |
|---|---|---:|---:|---:|
| RF | Độ sâu 16, 300 cây | 0,3992 | 0,4639 | 0,6544 |
| XGBoost | Độ sâu 8, 600 cây, learning rate 0,03 | 0,2675 | 0,3690 | 0,5794 |

XGBoost có MAE thấp hơn RF trên cả validation và test. Persistence trên cùng test đạt MAE 0,5841°C. Đây là hai model tốt nhất trong các cấu hình đã hoàn tất, chưa phải kết quả của toàn bộ dải 219 cấu hình.

Notebook đã có bảng kết quả và biểu đồ. Xem [báo cáo chốt model](../artifacts/ml_sliding_window/run_x2dflq_j/selected_models_report.md), [điểm test](../artifacts/ml_sliding_window/run_x2dflq_j/test_metrics.csv) và [điểm validation](../artifacts/ml_sliding_window/run_x2dflq_j/validation_metrics.csv). Không lưu file model.

## Kết quả lượt chạy 17/10/2026

Đây là lượt chạy cũ chọn một model tốt nhất chung, trước lượt 22/10/2026 và grid gọn 6 cấu hình. Kết quả hiện tại của grid 6 cấu hình nằm trong phần chia tập và báo cáo lượt chạy đầy đủ phía trên.

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

Model giảm MAE khoảng 35,8% so với persistence trên cùng giờ test. Toàn bộ model, dự đoán và biểu đồ của lượt notebook gọi thư viện trực tiếp nằm trong [thư mục kết quả](../artifacts/ml_sliding_window/20261017_223841_notebook_5p5kcsof/). Cả 20 code cell đã chạy thành công trong lượt lịch sử này; notebook hiện tại có 19 code cell và lưu kết quả của grid 6 cấu hình.

Test 2026 đã được xem ở các thử nghiệm cũ; không thể xem đây là tập hoàn toàn chưa từng được sử dụng. Dữ liệu ERA5 là tái phân tích: chất lượng trên backtest này không tự chứng minh chất lượng realtime.

Tham khảo: [scikit-learn về data leakage](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage), [feature trễ cho dự báo chuỗi thời gian](https://scikit-learn.org/1.7/auto_examples/applications/plot_time_series_lagged_features.html).
