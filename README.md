# Weather Forecasting — Hà Nội

Dự báo nhiệt độ một giờ tiếp theo bằng Random Forest, XGBoost, LSTM và GRU trên dữ liệu thời tiết theo giờ của Hà Nội.

## Luồng ML cửa sổ trượt: train đến 2024, validation 2025, test 2026

Dùng [train_model/ml_models.ipynb](train_model/ml_models.ipynb) và chạy các cell từ trên xuống. Đây là file duy nhất trong `train_model`. Có thể thực thi notebook bằng terminal:

```bash
jupyter nbconvert --to notebook --execute train_model/ml_models.ipynb --inplace
```

Notebook gọi trực tiếp các thư viện và chứa toàn bộ các cell xử lý/train/test. Input mặc định là hai CSV trong `data/processed`, hiện có dữ liệu đầy đủ; hai CSV raw trong bản tải này chỉ là con trỏ Git LFS. Sau khi khôi phục raw thật, có thể đổi `DATA_PATH` và `TEST_DATA_PATH` trong cell cấu hình để dùng raw.

- Cửa sổ 24 giờ, bước trượt 1 giờ; dự báo nhiệt độ sau 1 giờ.
- Chia theo **thời điểm nhãn**: train đến hết 2024, validation trong 2025, test các giờ có trong 2026.
- So sánh bốn cấu hình Random Forest/XGBoost bằng MAE validation; hòa thì dùng RMSE.
- Không dùng danh sách feature đã chọn ở lượt thử nghiệm cũ; tạo lại calendar và xác định cột hằng chỉ trên train.
- Lưu model đã chốt trước khi đọc test. Model đánh giá 2026 vẫn chỉ được fit trên train đến 2024; không refit trên 2025.

Kết quả CSV và biểu đồ nằm trong thư mục riêng cho mỗi lượt chạy tại `artifacts/ml_sliding_window/`. Hai model được giữ trong `selected_models` để test trực tiếp trong cùng kernel. Xem [hướng dẫn luồng mới](docs/ml_training.md).

**Các mục thử nghiệm feature bên dưới mô tả luồng cũ với holdout 20% và refit 2010–2025; đó là thử nghiệm riêng, không phải quy trình của notebook ML mới.**

## Cấu trúc

```text
crawl_data/                          # Thu thập ERA5 từ Open-Meteo
data/raw/Hanoi.csv                  # 2010–2025
data/raw/test/Hanoi_test_set.csv     # 01/01–29/09/2026
data/processed/Hanoi_final.csv      # Feature đã tạo, thông tin và nhiệt độ hiện tại
data/processed/Hanoi_test_final.csv # Schema giống train, có lịch sử qua đầu năm
src/weather_data.py                # Schema, kiểm tra giờ và feature nhân quả
src/feature_experiments.py          # CV, drop-column, bootstrap, đánh giá và biểu đồ
src/inspect_data.py                 # Kiểm tra raw bằng terminal
data_processing/clean_raw.ipynb
data_processing/prepare_test_set.ipynb
data_processing/feature_selection.json
data_processing/feature_experiment_review.md
eda/                               # EDA trên raw
train_model/ml_models.ipynb
docs/ml_training.md                # Hướng dẫn notebook ML
artifacts/                         # Model, predictions, metrics và biểu đồ
tests/
```

## Feature được bổ sung

| Nhóm | Feature | Cách tạo |
|---|---|---|
| Nhiệt độ hiện tại | `temperature_2m` | Giá trị quan sát tại giờ t; nhãn là nhiệt độ t+1. |
| Chu kỳ giờ | `hour_sin`, `hour_cos` | Sin/cos chu kỳ 24 giờ theo giờ Hà Nội. |
| Chu kỳ năm | `day_of_year_sin`, `day_of_year_cos` | Sin/cos chu kỳ 365 hoặc 366 ngày, gồm phần giờ trong ngày. |
| Lịch sử | `temp_lag_1h`, `temp_lag_3h`, `temp_lag_24h` | Nhiệt độ tại t−1, t−3, t−24. |
| Xu hướng | `temp_delta_1h` | Nhiệt độ tại t trừ t−1. |
| Trung bình gần nhất | `temp_rolling_mean_3h` | Trung bình t−2, t−1, t; yêu cầu đủ ba quan sát. |
| Hướng gió | `wind_direction_10m_sin/cos`, `wind_direction_100m_sin/cos` | Chuyển góc độ sang sin/cos; 359° và 1° nằm gần nhau. |

Raw đã có `relative_humidity_2m`, `rain`, `wind_speed_10m` cùng các biến khí tượng còn lại; không cần tải thêm. Giữ `apparent_temperature`, ET0, mây, nhiệt độ đất và các nhóm khác để thử nghiệm. Raw luôn giữ các hướng gió gốc; model dùng cặp sin/cos.

Không dùng tương lai để tạo feature. Chỉ forward-fill biến đầu vào tối đa ba giờ; nhiệt độ mục tiêu không được điền hoặc nội suy. Dữ liệu phải liên tục từng giờ, không có timestamp trùng hay giá trị không hữu hạn.

Train bỏ đúng 24 giờ đầu thiếu lịch sử: processed train có 140.232 dòng, bắt đầu 02/01/2010 00:00. Test giữ đủ 6.528 giờ nhờ dùng 24 giờ cuối raw train làm history_data. Dataset processed giữ hợp của danh sách input ML và DL cộng bốn cột thông tin, vì hai họ model có thể dùng bộ feature khác nhau. Nhãn tương lai được tạo khi chuẩn bị train, không lưu sẵn trong processed.

## Chọn feature và kiểm tra biến trùng lặp

Thử nghiệm so sánh bốn bước trên cùng tập giờ dự báo và cấu hình model:

1. `legacy_weather`: bộ biến khí tượng cũ, input một giờ.
2. `weather_cyclic_wind`: đổi hướng gió sang sin/cos.
3. `weather_time`: bổ sung chu kỳ giờ và năm.
4. `all_augmented`: bổ sung lag, delta và rolling nhiệt độ.

Cả RF/XGBoost đều được fit lại khi bỏ từng cột `vapour_pressure_deficit`, `dew_point_2m`, `is_day`, `soil_temperature_0_to_7cm`, `soil_temperature_7_to_28cm`; có thêm thử nghiệm bỏ cặp nhiệt độ đất và cả nhóm này. Importance từ cây được lưu để diễn giải, không dùng riêng nó để xóa biến. Cột hằng được xác định trên phần phát triển; dùng `cloud_cover` đại diện mây như pipeline trước.

20% thời gian cuối 2010–2025 làm holdout, bắt đầu **20/10/2022 04:00** theo giờ của nhãn. Trên phần phát triển: 5 fold `TimeSeriesSplit`, gap bằng horizon, mỗi fold dùng tối đa bốn năm train gần nhất và toàn bộ validation. Cấu hình so sánh cố định: RF 60 cây/depth 12; XGBoost 200 cây/depth 5. CV này đánh giá input một giờ với lag tường minh; không phải so sánh trực tiếp với lượt ML cửa sổ 24 giờ trước đây.

Quy tắc loại biến được định trước:

- So sánh sai số trên đúng cùng giờ dự báo; bootstrap ghép cặp theo ngày, 2.000 lần.
- Cận trên một phía của mức tăng MAE **và** RMSE phải trong 1% cho **cả** RF/XGBoost. Hiệu chỉnh Bonferroni cho các thử nghiệm, hai họ model và hai chỉ số.
- Không có nhóm mùa hoặc giờ tăng MAE/RMSE quá 5% trên dự đoán CV. Đây là kiểm tra nhóm mô tả, tách khỏi confidence bounds tổng thể.
- Các cột đạt riêng lẻ được thử bỏ chung; chỉ áp dụng bộ gọn nếu thử nghiệm chung cũng đạt. Nếu chưa đủ bằng chứng thì giữ biến.

1% và 5% là mức suy giảm thực dụng cho dự án, không phải định nghĩa phổ quát về ý nghĩa thống kê. Không đồng nhất “không có khác biệt có ý nghĩa” với “đã chứng minh tương đương”. Bootstrap theo ngày chưa mô hình hóa tương quan qua nhiều ngày. CV dùng để phát triển/chọn feature, không phải nested CV.

Bộ feature và model được chốt bằng CV trước khi đánh giá holdout và mở raw 2026. Các bảng holdout/2026 không dùng để chọn lại feature hoặc tham số. Khi đánh giá 2026, model được refit trên toàn bộ 2010–2025; các quan sát test đã có tại thời điểm dự báo được dùng làm lịch sử cho những giờ sau. Đây là dự báo từng giờ, không phải dự báo cả năm từ một mốc duy nhất.

## Kết quả và các file lưu

Lượt thử nghiệm đã chốt **33 feature ML** (`joint_reduced`) và **XGBoost** bằng CV. Bỏ `dew_point_2m`, `vapour_pressure_deficit`, `is_day`, `soil_temperature_7_to_28cm` khỏi input ML sau kiểm tra riêng và bỏ chung; giữ tầng đất 0–7 cm. DL mặc định giữ **32 feature weather + time**. Processed có 41 cột vì vẫn giữ các biến này cho DL.

Trên test 2026, XGBoost đã chốt đạt **MAE 0,3887°C**. Bộ khí tượng cũ dùng input một giờ đạt 0,4596°C trong cùng lượt so sánh; giảm MAE khoảng 15,4%. Đây chưa phải so sánh với model cửa sổ 24 giờ cũ hoặc kết quả LSTM/GRU mới.

Đọc [báo cáo thử nghiệm](data_processing/feature_experiment_review.md) để xem RMSE, kết quả theo mùa/giờ và từng thử nghiệm loại biến.

`data_processing/feature_selection.json` lưu:

- `selected_features`, `active_feature_set`: danh sách ML theo đúng thứ tự input.
- `dl_selected_features`: bộ weather + time cho chuỗi 24 giờ; giữ các biến khí tượng để đánh giá riêng cho DL.
- `feature_sets`, `strategy_cv_scores`, `strategy_fold_scores`, `fold_periods`: so sánh các nhóm và biên thời gian mỗi fold.
- `drop_column_scores`, `joint_drop_scores`: delta sai số, confidence bounds, kiểm tra theo mùa/giờ và quyết định loại.
- `feature_scores`: importance của RF/XGBoost; không phải quy tắc xóa biến.
- `feature_engineering`, cấu hình thử nghiệm, mốc holdout và SHA-256 của raw/processed.
- `holdout_scores`, `external_test_scores`, `evaluation_slices`: kết quả trên các tập đánh giá sau khi chốt.

`artifacts/feature_experiments/` có các CSV của CV, drop-column và evaluation, dự đoán từng giờ, `metrics_by_season_hour.csv`, hai biểu đồ PNG, và model đã refit để đánh giá 2026. Các mùa được định nghĩa DJF/MAM/JJA/SON; nhóm giờ và mùa dựa trên **thời điểm nhãn** theo giờ Hà Nội. Test 2026 chỉ có tháng 1–2 cho DJF và tháng 9 cho SON; hai mùa này chưa đủ cả ba tháng.

Model được chọn bằng CV của lượt thử nghiệm được lưu thêm trong `artifacts/ml/best_ml_feature_experiment_*.joblib`, cùng danh sách feature và cấu hình. Notebook đánh giá 2026 có thể đọc model này. Các model cũ được giữ; chúng có thể không khớp schema mới, và notebook sẽ báo lỗi thay vì tự ép schema.

## Cách chạy

Dùng Python 3.12; môi trường dự án ghim các phiên bản trong requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Để tạo lại toàn bộ feature, chạy selection, đánh giá holdout/2026 và lưu kết quả:

```bash
python -m src.feature_experiments
```

Lệnh này thực hiện đầy đủ bốn ưu tiên. Có thể chạy `data_processing/clean_raw.ipynb` để làm cùng quy trình và xem bảng trong notebook. Nếu chỉ cần tạo lại processed test sau khi đã có selection/train:

```bash
jupyter nbconvert --to notebook --execute data_processing/prepare_test_set.ipynb --inplace
```

Các bước tiếp theo:

| File | Công việc |
|---|---|
| `eda/data_analyze.ipynb`, `eda/eda_visualization.ipynb` | EDA đọc raw, chạy độc lập. |
| `train_model/ml_models.ipynb` | Cửa sổ 24 giờ; train đến 2024, chọn RF/XGBoost bằng validation 2025 và test 2026. |

ML mặc định `SEQUENCE_LENGTH = 1` vì lag 1/3/24 giờ, delta và rolling đã được tạo tường minh. Có thể thử lại cửa sổ 24 giờ; khi đó mỗi feature được mở rộng thêm 23 lag và cần đánh giá cấu hình mới bằng CV. Không mặc định coi input 24 giờ và input một giờ là cùng một thử nghiệm.

DL mặc định `DL_FEATURE_SET = "weather_time"`, ưu tiên feature thời gian và giữ nguyên chuỗi 24 giờ, không đưa mọi lag nhiệt độ thủ công vào. Có thể đổi sang `weather_cyclic_wind` hoặc `all_augmented` để so sánh riêng bằng validation với cùng split/seed/kiến trúc. Kết quả loại biến của RF/XGBoost không được áp dụng tự động cho LSTM/GRU. Các chỉ số trong báo cáo feature là kết quả ML; cần chạy notebook DL để có kết quả huấn luyện DL mới.

Để chạy DL:

```bash
pip install -r requirements-dl.txt
python -m ipykernel install --user --name weather-dl --display-name weather-dl
```

Nếu cần tải lại raw:

```bash
python crawl_data/crawl_hanoi_enriched_weather.py
python crawl_data/crawl_test_set.py
python src/inspect_data.py
```

Hai script thu thập dùng cùng nguồn ERA5, tọa độ, múi giờ UTC+7 và đơn vị. Chỉ thay CSV khi dữ liệu đầy đủ; lỗi API giữ file cũ. Raw test đã được tải lại từ ERA5 trong lượt trước của dự án; không dùng forecast API để điền nhãn test.

## Kiểm tra và giới hạn

```bash
python -m unittest discover -s tests -v
```

Kiểm tra bao gồm giá trị lag/delta/rolling, dữ liệu tương lai không làm đổi input quá khứ, năm nhuận và chu kỳ năm, hướng gió 359°/1°, history_data đầu năm, nhãn đúng horizon, biên train/validation/test, scaler, SHA-256, quyết định loại biến và một lượt pipeline hoàn chỉnh trên dữ liệu nhỏ.

MAE/RMSE theo °C là chỉ số chính; MAPE trên thang °C chỉ để tham khảo. Đây là backtest trên ERA5 tái phân tích, chưa chứng minh chất lượng dự báo realtime. Test 2026 từng được đánh giá trong các lượt trước, nên không còn là tập hoàn toàn chưa ai xem; pipeline mới không dùng nó để chọn feature. Không dùng điểm các tập đánh giá để điều chỉnh lựa chọn sau khi đọc báo cáo.

Phương pháp chu kỳ thời gian và chia dữ liệu tham khảo tài liệu chính thức [scikit-learn về feature thời gian](https://scikit-learn.org/1.7/auto_examples/applications/plot_cyclical_feature_engineering.html), [TimeSeriesSplit](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) và [tránh rò rỉ dữ liệu](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage).
