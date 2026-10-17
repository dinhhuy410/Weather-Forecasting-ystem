# Kết quả thử nghiệm feature

Mã lượt chạy theo đồng hồ hệ thống: 20261013_092300 (Asia/Ho_Chi_Minh).
Bộ ML chốt bằng CV: **joint_reduced**, 33 feature.
Model chốt bằng CV: **XGBoost**.
Các lựa chọn này được cố định trước khi mở test 2026; holdout/test chỉ dùng để đánh giá.

## Phương pháp

5 fold thời gian, gap bằng horizon 1 giờ. Mỗi fold dùng tối đa 4 năm train gần nhất; validation giữ đầy đủ.
RF: 60 cây, depth 12; XGBoost: 200 cây, depth 5. Cấu hình cố định, chưa tuning lại.
MAE/RMSE tính bằng °C. So sánh loại cột trên cùng giờ dự báo, bootstrap 2.000 lần theo ngày.
Chỉ bỏ cột nếu cận trên suy giảm nằm trong 1% cho cả MAE/RMSE, cả RF/XGB, với hiệu chỉnh Bonferroni;
đồng thời không có nhóm mùa/giờ tăng sai số quá 5%. Các cột được bỏ cùng nhau phải vượt qua kiểm tra chung.
Bootstrap theo ngày chưa mô hình hóa tương quan qua nhiều ngày; CV phát triển không phải nested CV.

## Sai số trên dữ liệu đánh giá

| Tập | Bộ feature | Model | MAE | RMSE |
|---|---|---|---:|---:|
| holdout | legacy_weather | Random Forest | 0.4429 | 0.6387 |
| holdout | legacy_weather | XGBoost | 0.4451 | 0.6288 |
| holdout | weather_cyclic_wind | Random Forest | 0.4412 | 0.6372 |
| holdout | weather_cyclic_wind | XGBoost | 0.4448 | 0.6272 |
| holdout | weather_time | Random Forest | 0.3178 | 0.4583 |
| holdout | weather_time | XGBoost | 0.3066 | 0.4371 |
| holdout | all_augmented | Random Forest | 0.3191 | 0.4604 |
| holdout | all_augmented | XGBoost | 0.3071 | 0.4369 |
| holdout | joint_reduced | Random Forest | 0.3194 | 0.4618 |
| holdout | joint_reduced | XGBoost | 0.3100 | 0.4389 |
| holdout | persistence | Persistence baseline | 0.5627 | 0.9133 |
| test_2026 | legacy_weather | Random Forest | 0.4539 | 0.6753 |
| test_2026 | legacy_weather | XGBoost | 0.4596 | 0.6685 |
| test_2026 | weather_cyclic_wind | Random Forest | 0.4541 | 0.6761 |
| test_2026 | weather_cyclic_wind | XGBoost | 0.4606 | 0.6708 |
| test_2026 | weather_time | Random Forest | 0.3987 | 0.6204 |
| test_2026 | weather_time | XGBoost | 0.3894 | 0.6001 |
| test_2026 | all_augmented | Random Forest | 0.3969 | 0.6221 |
| test_2026 | all_augmented | XGBoost | 0.3881 | 0.6007 |
| test_2026 | joint_reduced | Random Forest | 0.3980 | 0.6237 |
| test_2026 | joint_reduced | XGBoost | 0.3887 | 0.5985 |
| test_2026 | persistence | Persistence baseline | 0.5841 | 0.9396 |

## Drop-column

Delta dương nghĩa là bỏ biến làm tăng sai số.

| Biến/nhóm bỏ | Model | ΔMAE | ΔRMSE | Cận trên MAE | Đủ điều kiện bỏ |
|---|---|---:|---:|---:|---|
| drop_vapour_pressure_deficit | Random Forest | -0.00001 | 0.00015 | 0.00031 | Có |
| drop_vapour_pressure_deficit | XGBoost | 0.00120 | 0.00223 | 0.00207 | Có |
| drop_dew_point_2m | Random Forest | -0.00007 | -0.00001 | 0.00023 | Có |
| drop_dew_point_2m | XGBoost | 0.00082 | 0.00189 | 0.00173 | Có |
| drop_is_day | Random Forest | 0.00002 | 0.00022 | 0.00029 | Có |
| drop_is_day | XGBoost | 0.00099 | 0.00237 | 0.00185 | Có |
| drop_soil_temperature_0_to_7cm | Random Forest | 0.00036 | 0.00292 | 0.00178 | Không |
| drop_soil_temperature_0_to_7cm | XGBoost | 0.00069 | 0.00143 | 0.00151 | Có |
| drop_soil_temperature_7_to_28cm | Random Forest | 0.00022 | 0.00026 | 0.00066 | Có |
| drop_soil_temperature_7_to_28cm | XGBoost | 0.00104 | 0.00252 | 0.00197 | Có |
| drop_soil_pair | Random Forest | 0.00072 | 0.00298 | 0.00226 | Không |
| drop_soil_pair | XGBoost | 0.00103 | 0.00245 | 0.00207 | Không |
| drop_redundant_group | Random Forest | 0.00042 | 0.00247 | 0.00202 | Không |
| drop_redundant_group | XGBoost | 0.00169 | 0.00377 | 0.00277 | Không |

Các biến đã bỏ: dew_point_2m, vapour_pressure_deficit, soil_temperature_7_to_28cm, is_day

LSTM/GRU mặc định dùng weather + time trong chuỗi 24 giờ, giữ toàn bộ biến khí tượng;
không dùng các lag/delta/rolling nhiệt độ thủ công. Kết luận ablation ML không được áp dụng tự động cho DL.
Notebook DL có lựa chọn bộ input để chạy so sánh riêng; bảng trên là kết quả RF/XGBoost.

## File kết quả

- `feature_selection.json`: bộ chốt, cấu hình, confidence bounds, chi tiết theo mùa/giờ và SHA-256.
- `../artifacts/feature_experiments/cv_summary.csv`: so sánh từng nhóm feature bằng CV.
- `../artifacts/feature_experiments/drop_column_importance.csv`: kết quả loại cột.
- `../artifacts/feature_experiments/metrics_by_season_hour.csv`: MAE/RMSE và số mẫu mỗi mùa/giờ.
- `../artifacts/feature_experiments/evaluation_predictions.csv`: dự đoán ghép cặp để kiểm tra lại.
- `../artifacts/feature_experiments/selected_models_refit_2026.joblib`: model được refit toàn bộ 2010–2025.

Test 2026: DJF chỉ có tháng 1–2, SON chỉ có tháng 9; hai mùa này chưa đủ cả ba tháng.
Test 2026 đã được đánh giá trong các lượt trước của dự án, vì vậy không còn là lockbox hoàn toàn mới.

![So sánh feature](../artifacts/feature_experiments/feature_comparison.png)

![Sai số theo mùa và giờ](../artifacts/feature_experiments/errors_by_season_hour.png)
