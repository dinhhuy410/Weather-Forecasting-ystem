"""Chronological feature experiments, paired daily uncertainty and error slices."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.weather_data import (
    WEATHER_COLUMNS, METADATA_COLUMNS, TIME_FEATURES, HISTORY_FEATURES,
    WIND_DIRECTION_COLUMNS, FEATURE_ENGINEERING_VERSION,
    engineer_forecast_features, validate_hourly_frame, build_forecast_dataset,
)

MODELS = ("Random Forest", "XGBoost")
REDUNDANCY_COLUMNS = [
    "vapour_pressure_deficit", "dew_point_2m", "is_day",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
]


@dataclass(frozen=True)
class ExperimentConfig:
    horizon_hours: int = 1
    holdout_fraction: float = 0.2
    cv_splits: int = 5
    random_state: int = 42
    max_train_hours: int = 4 * 366 * 24
    rf_trees: int = 60
    xgb_trees: int = 200
    n_jobs: int = 4
    noninferiority_margin_pct: float = 1.0
    segment_margin_pct: float = 5.0
    bootstrap_repeats: int = 2000


def read_clean_weather(path):
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame = frame.drop_duplicates().sort_values("time").reset_index(drop=True)
    for column in frame.columns:
        if column not in ("time", "city"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["temperature_2m"].isna().any():
        raise ValueError("Missing target measurements; do not fabricate labels.")
    fill = [c for c in frame if c not in ("time", "temperature_2m")]
    frame[fill] = frame[fill].ffill(limit=3)
    validate_hourly_frame(frame, [*WEATHER_COLUMNS, *METADATA_COLUMNS])
    return frame


def make_model(name, config, seed):
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=config.rf_trees, max_depth=12, min_samples_leaf=2,
            random_state=seed, n_jobs=config.n_jobs,
        )
    if name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror", n_estimators=config.xgb_trees,
            max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, random_state=seed, n_jobs=config.n_jobs,
            tree_method="hist", eval_metric="rmse",
        )
    raise ValueError(f"Unknown model: {name}")


def prediction_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def stratified_metrics(label_times, actual, predicted):
    """Group by target hour/meteorological season, in Hanoi local time."""
    times = pd.Series(pd.to_datetime(np.asarray(label_times))).reset_index(drop=True)
    if times.dt.tz is not None:
        times = times.dt.tz_convert("Asia/Ho_Chi_Minh")
    frame = pd.DataFrame({"actual": np.asarray(actual), "predicted": np.asarray(predicted)})
    frame["hour"] = times.dt.hour
    frame["season"] = times.dt.month.map({
        12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON",
    })
    rows = []
    for group in ("season", "hour"):
        for value, part in frame.groupby(group, sort=True):
            rows.append({"group": group, "value": str(value), "n": len(part),
                         **prediction_metrics(part.actual, part.predicted)})
    return rows


def paired_daily_comparison(times, actual, baseline, candidate, *, repeats=2000, seed=42, confidence=0.95):
    """Bootstrap paired calendar days, preserving within-day error dependence.

    Positive delta means the candidate is worse. Upper one-sided bounds must
    fit inside a predeclared practical margin; lack of significance alone is
    never evidence that a feature can be removed.
    """
    actual, baseline, candidate = map(lambda x: np.asarray(x, dtype=float), (actual, baseline, candidate))
    if not (len(times) == len(actual) == len(baseline) == len(candidate)) or len(actual) < 2:
        raise ValueError("Paired comparisons need aligned observations.")
    frame = pd.DataFrame({
        "day": pd.to_datetime(np.asarray(times)).normalize(),
        "base_abs": np.abs(actual - baseline), "candidate_abs": np.abs(actual - candidate),
        "base_sq": (actual - baseline) ** 2, "candidate_sq": (actual - candidate) ** 2,
    })
    frame["n"] = 1
    blocks = frame.groupby("day").sum(numeric_only=True).to_numpy()
    rng = np.random.default_rng(seed)
    deltas = []
    # Small chunks keep bootstrap memory independent of the number of repeats.
    for start in range(0, repeats, 100):
        indices = rng.integers(len(blocks), size=(min(100, repeats - start), len(blocks)))
        sums = blocks[indices].sum(axis=1)
        deltas.append(np.column_stack([
            (sums[:, 1] - sums[:, 0]) / sums[:, 4],
            np.sqrt(sums[:, 3] / sums[:, 4]) - np.sqrt(sums[:, 2] / sums[:, 4]),
        ]))
    samples = np.concatenate(deltas)
    base_metrics, candidate_metrics = prediction_metrics(actual, baseline), prediction_metrics(actual, candidate)
    return {
        "n": len(actual), "daily_blocks": len(blocks), "confidence": confidence,
        "bootstrap_repeats": repeats,
        **{f"delta_{metric}": candidate_metrics[metric] - base_metrics[metric] for metric in ("MAE", "RMSE")},
        **{f"{metric}_baseline": base_metrics[metric] for metric in ("MAE", "RMSE")},
        **{f"{metric}_upper_bound": float(np.quantile(samples[:, i], confidence)) for i, metric in enumerate(("MAE", "RMSE"))},
        **{f"{metric}_ci_low": float(np.quantile(samples[:, i], 0.025)) for i, metric in enumerate(("MAE", "RMSE"))},
        **{f"{metric}_ci_high": float(np.quantile(samples[:, i], 0.975)) for i, metric in enumerate(("MAE", "RMSE"))},
    }


def safe_to_remove(comparison, slices, config):
    global_ok = all(
        comparison[f"{m}_upper_bound"] <= comparison[f"{m}_baseline"] * config.noninferiority_margin_pct / 100
        for m in ("MAE", "RMSE")
    )
    # Slice guard is descriptive, separate from the simultaneous global bounds.
    segment_ok = all(row["relative_degradation_pct"] <= config.segment_margin_pct for row in slices)
    return bool(global_ok and segment_ok)


def compare_slices(times, actual, baseline, candidate):
    base = stratified_metrics(times, actual, baseline)
    new = stratified_metrics(times, actual, candidate)
    rows = []
    for b, c in zip(base, new):
        for metric in ("MAE", "RMSE"):
            delta = c[metric] - b[metric]
            rows.append({"group": b["group"], "value": b["value"], "metric": metric,
                         "n": b["n"], "baseline": b[metric], "candidate": c[metric],
                         "delta": delta, "relative_degradation_pct": float(100 * delta / max(b[metric], 1e-12))})
    return rows


def feature_sets_for_experiment(frame, holdout_start):
    # Only development observations determine constant weather columns.
    excluded = {"cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"}
    weather = [c for c in WEATHER_COLUMNS if c not in excluded]
    dev = frame[frame.time < holdout_start]
    constants = [c for c in weather if dev[c].nunique(dropna=False) <= 1]
    weather = [c for c in weather if c not in constants]
    encoded = []
    for c in weather:
        encoded.extend([f"{c}_sin", f"{c}_cos"] if c in WIND_DIRECTION_COLUMNS else [c])
    full = [*encoded, *TIME_FEATURES, *HISTORY_FEATURES]
    sets = {
        "legacy_weather": weather,
        "weather_cyclic_wind": encoded,
        "weather_time": [*encoded, *TIME_FEATURES],
        "all_augmented": full,
    }
    drops = {f"drop_{c}": [c] for c in REDUNDANCY_COLUMNS if c in full}
    drops["drop_soil_pair"] = [c for c in REDUNDANCY_COLUMNS if c.startswith("soil_") and c in full]
    drops["drop_redundant_group"] = [c for c in REDUNDANCY_COLUMNS if c in full]
    for group, columns in drops.items():
        sets[group] = [c for c in full if c not in columns]
    return sets, drops, constants


def run_feature_experiments(project_root, config=None, progress=print):
    """Select using development OOF predictions, then evaluate frozen sets."""
    root = Path(project_root)
    config = config or ExperimentConfig()
    raw_path = root / "data/raw/Hanoi.csv"
    raw = read_clean_weather(raw_path)
    holdout_start = raw.time.iloc[int(len(raw) * (1 - config.holdout_fraction))]
    engineered = engineer_forecast_features(raw)
    sets, drops, constants = feature_sets_for_experiment(engineered, holdout_start)
    all_columns = list(dict.fromkeys(c for columns in sets.values() for c in columns))
    dataset, label = build_forecast_dataset(engineered, all_columns, "temperature_2m", config.horizon_hours)
    dataset[all_columns] = dataset[all_columns].astype(np.float32)
    development = dataset[dataset.label_time < holdout_start].reset_index(drop=True)
    holdout = dataset[dataset.label_time >= holdout_start].reset_index(drop=True)
    folds = list(TimeSeriesSplit(n_splits=config.cv_splits, gap=config.horizon_hours,
                                max_train_size=config.max_train_hours).split(development))
    rows, fold_periods, oof, importance = [], [], {}, {name: [] for name in MODELS}
    for fold, (train_idx, valid_idx) in enumerate(folds, 1):
        train, valid = development.iloc[train_idx], development.iloc[valid_idx]
        if train.label_time.max() > valid.time.min():
            raise ValueError("CV training labels extend beyond validation input time.")
        fold_periods.append({"fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                             "train_label_end": str(train.label_time.max()),
                             "valid_label_start": str(valid.label_time.min()),
                             "valid_label_end": str(valid.label_time.max())})
        for set_name, columns in sets.items():
            for model_name in MODELS:
                model = make_model(model_name, config, config.random_state + fold)
                model.fit(train[columns], train[label])
                prediction = model.predict(valid[columns])
                key = (set_name, model_name)
                oof.setdefault(key, []).append(pd.DataFrame({"label_time": valid.label_time.to_numpy(),
                    "actual": valid[label].to_numpy(), "predicted": prediction, "fold": fold}))
                metrics = prediction_metrics(valid[label], prediction)
                rows.append({"feature_set": set_name, "Model": model_name, "fold": fold,
                             "k": len(columns), **metrics})
                if set_name == "all_augmented":
                    importance[model_name].append(model.feature_importances_)
                progress(f"CV {fold}/{config.cv_splits} {set_name} {model_name}: MAE={metrics['MAE']:.4f}")

    def assess(set_name, *, confidence):
        comparisons = []
        for model_name in MODELS:
            base = pd.concat(oof[("all_augmented", model_name)], ignore_index=True)
            candidate = pd.concat(oof[(set_name, model_name)], ignore_index=True)
            pd.testing.assert_series_equal(base.label_time, candidate.label_time)
            comparison = paired_daily_comparison(base.label_time, base.actual, base.predicted, candidate.predicted,
                repeats=config.bootstrap_repeats, seed=config.random_state, confidence=confidence)
            slices = compare_slices(base.label_time, base.actual, base.predicted, candidate.predicted)
            comparisons.append({"feature_set": set_name, "Model": model_name,
                **comparison, "slice_comparisons": slices,
                "safe_to_remove": safe_to_remove(comparison, slices, config)})
        return comparisons

    # Bonferroni protects the two metrics, two model families and all tested drops.
    confidence = 1 - 0.05 / (len(drops) * len(MODELS) * 2)
    ablations = [row for set_name in drops for row in assess(set_name, confidence=confidence)]
    safe_columns = [c for c in REDUNDANCY_COLUMNS if f"drop_{c}" in drops and
                    all(row["safe_to_remove"] for row in ablations if row["feature_set"] == f"drop_{c}")]
    active = "all_augmented"
    joint_comparisons = []
    if safe_columns:
        sets["joint_reduced"] = [c for c in sets["all_augmented"] if c not in safe_columns]
        # Individual safe drops can interact: retrain and verify the union.
        for fold, (train_idx, valid_idx) in enumerate(folds, 1):
            train, valid = development.iloc[train_idx], development.iloc[valid_idx]
            for model_name in MODELS:
                columns = sets["joint_reduced"]
                model = make_model(model_name, config, config.random_state + fold)
                model.fit(train[columns], train[label])
                predicted = model.predict(valid[columns])
                oof.setdefault(("joint_reduced", model_name), []).append(pd.DataFrame({
                    "label_time": valid.label_time.to_numpy(), "actual": valid[label].to_numpy(),
                    "predicted": predicted, "fold": fold}))
                rows.append({"feature_set": "joint_reduced", "Model": model_name, "fold": fold,
                             "k": len(columns), **prediction_metrics(valid[label], predicted)})
                progress(f"CV {fold}/{config.cv_splits} joint_reduced {model_name}")
        joint_comparisons = assess("joint_reduced", confidence=0.9875)
        if all(row["safe_to_remove"] for row in joint_comparisons):
            active = "joint_reduced"
    selected = sets[active]
    progress(f"Frozen feature set before holdout/test: {active} ({len(selected)} features)")

    output = root / "train_model/artifacts/feature_experiments"
    output.mkdir(parents=True, exist_ok=True)
    cv_results = pd.DataFrame(rows)
    summary = cv_results.groupby(["feature_set", "Model"], as_index=False).agg(
        k=("k", "first"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), R2_mean=("R2", "mean"))
    selected_cv = summary[summary.feature_set == active]
    cv_mae = dict(zip(selected_cv.Model, selected_cv.MAE_mean))
    best_model = min(cv_mae, key=cv_mae.get)
    cv_results.to_csv(output / "cv_fold_scores.csv", index=False)
    summary.to_csv(output / "cv_summary.csv", index=False)
    all_oof = []
    for (set_name, model_name), parts in oof.items():
        result = pd.concat(parts, ignore_index=True)
        result["feature_set"], result["Model"] = set_name, model_name
        all_oof.append(result)
    pd.concat(all_oof, ignore_index=True).to_csv(output / "cv_predictions.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "slice_comparisons"} for row in ablations]).to_csv(
        output / "drop_column_importance.csv", index=False)
    importance_rows = []
    for model_name in MODELS:
        values = np.vstack(importance[model_name])
        for i, c in enumerate(sets["all_augmented"]):
            importance_rows.append({"Model": model_name, "feature": c,
                "importance_mean": float(values[:, i].mean()), "importance_std": float(values[:, i].std()),
                "selected": c in selected})

    processed = root / "data/processed/Hanoi_final.csv"
    processed.parent.mkdir(parents=True, exist_ok=True)
    # Store DL candidates too; the ML decision does not establish DL redundancy.
    dl_selected = sets["weather_time"]
    saved_features = list(dict.fromkeys([*selected, *dl_selected]))
    engineered[[*METADATA_COLUMNS, *saved_features]].to_csv(processed, index=False)
    report = {
        "target": "temperature_2m", "horizon_hours": config.horizon_hours,
        "cv_splits": config.cv_splits, "holdout_fraction": config.holdout_fraction,
        "random_state": config.random_state, "holdout_start": str(holdout_start),
        "selection_train_label_end": str(development.label_time.max()),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "processed_sha256": hashlib.sha256(processed.read_bytes()).hexdigest(),
        "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
        "feature_engineering": {"time_features": TIME_FEATURES, "history_features": HISTORY_FEATURES,
            "temperature_lags_hours": [1, 3, 24], "rolling_windows_hours": [3],
            "rolling_includes_current": True, "warmup_hours": 24, "timezone": "Asia/Ho_Chi_Minh",
            "wind_encoding": "sin_cos", "year_period": "365_or_366_days"},
        "active_feature_set": active, "selected_features": selected,
        "dl_selected_features": dl_selected,
        "dl_selection_policy": "retain_weather_and_time; no_manual_temperature_history_in_24h_sequences",
        "selection_policy": "paired_daily_noninferiority_on_development_cv_then_joint_verification",
        "noninferiority": {"margin_pct": config.noninferiority_margin_pct,
            "segment_margin_pct": config.segment_margin_pct, "simultaneous_confidence": confidence,
            "correction": "Bonferroni over drop candidates, model families and metrics",
            "block": "calendar_day", "bootstrap_repeats": config.bootstrap_repeats,
            "limitations": "Daily blocks do not model multi-day dependence; CV is development, not nested."},
        "experiment_config": asdict(config), "constant_features": constants,
        "feature_sets": {name: {"features": columns, "feature_count": len(columns)} for name, columns in sets.items()},
        "strategy_cv_scores": summary.to_dict(orient="records"),
        "strategy_fold_scores": rows, "fold_periods": fold_periods,
        "feature_scores": importance_rows, "drop_column_scores": ablations,
        "joint_drop_scores": joint_comparisons,
        "removed_features": [c for c in sets["all_augmented"] if c not in selected],
        "evaluation_policy": "freeze features and model before holdout or external test; no retuning from evaluation",
        "selected_model_by_cv": best_model,
    }
    report_path = root / "data_processing/feature_selection.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Raw 2026 is first opened after the choice has been frozen.
    external_raw_path = root / "data/raw/test/Hanoi_test_set.csv"
    external_raw = read_clean_weather(external_raw_path)
    external = engineer_forecast_features(external_raw, history=raw)
    external[[*METADATA_COLUMNS, *saved_features]].to_csv(root / "data/processed/Hanoi_test_final.csv", index=False)
    combined = pd.concat([engineered, external], ignore_index=True)
    combined_dataset, _ = build_forecast_dataset(combined, all_columns, "temperature_2m", config.horizon_hours)
    combined_dataset[all_columns] = combined_dataset[all_columns].astype(np.float32)
    external_train = combined_dataset[combined_dataset.label_time <= raw.time.max()].copy()
    external_test = combined_dataset[combined_dataset.label_time >= external.time.min()].copy()
    if len(external_test) != len(external):
        raise ValueError("External evaluation lost first labels or history at the train/test boundary.")
    timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S")
    eval_names = list(dict.fromkeys(["legacy_weather", "weather_cyclic_wind", "weather_time", "all_augmented", active]))
    evaluation_rows, slice_rows, prediction_rows = [], [], []
    dev_models, external_models = {}, {}
    for split, train, test in [("holdout", development, holdout), ("test_2026", external_train, external_test)]:
        for set_name in eval_names:
            columns = sets[set_name]
            for model_name in MODELS:
                model = make_model(model_name, config, config.random_state)
                model.fit(train[columns], train[label])
                predicted = model.predict(test[columns])
                evaluation_rows.append({"split": split, "feature_set": set_name, "Model": model_name,
                    "k": len(columns), "n": len(test), **prediction_metrics(test[label], predicted)})
                for row in stratified_metrics(test.label_time, test[label], predicted):
                    slice_rows.append({"split": split, "feature_set": set_name, "Model": model_name, **row})
                prediction_rows.append(pd.DataFrame({"split": split, "feature_set": set_name, "Model": model_name,
                    "time": test.time.to_numpy(), "label_time": test.label_time.to_numpy(),
                    "actual": test[label].to_numpy(), "predicted": predicted}))
                if set_name == active:
                    (dev_models if split == "holdout" else external_models)[model_name] = model
                progress(f"{split} {set_name} {model_name}: MAE={evaluation_rows[-1]['MAE']:.4f}")
        persistence = test.temperature_2m.to_numpy()
        evaluation_rows.append({"split": split, "feature_set": "persistence", "Model": "Persistence baseline",
                               "k": 1, "n": len(test), **prediction_metrics(test[label], persistence)})
        for row in stratified_metrics(test.label_time, test[label], persistence):
            slice_rows.append({"split": split, "feature_set": "persistence", "Model": "Persistence baseline", **row})
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(output / "evaluation_scores.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(output / "metrics_by_season_hour.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / "evaluation_predictions.csv", index=False)
    common_bundle = {"model_name": best_model, "feature_columns": selected, "base_feature_columns": selected,
        "label_column": label, "target": "temperature_2m", "horizon_hours": config.horizon_hours,
        "sequence_length": 1, "lags": [], "engineered_lags": [1, 3, 24], "rolling_windows": [3],
        "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
        "data_sha256": report["processed_sha256"], "selection_cv_mae": cv_mae}
    ml_dir = root / "train_model/artifacts/ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    model_file = ml_dir / f"best_ml_feature_experiment_temperature_2m_{timestamp}.joblib"
    joblib.dump({**common_bundle, "models": dev_models, "model": dev_models[best_model],
                "training_label_end": str(development.label_time.max())}, model_file, compress=3)
    joblib.dump({**common_bundle, "models": external_models, "model": external_models[best_model],
                "training_label_end": str(external_train.label_time.max())}, output / "selected_models_refit_2026.joblib", compress=3)
    report.update({"holdout_scores": [r for r in evaluation_rows if r["split"] == "holdout"],
        "external_test_scores": [r for r in evaluation_rows if r["split"] == "test_2026"],
        "evaluation_slices": slice_rows, "selected_model_by_cv": best_model,
        "external_raw_sha256": hashlib.sha256(external_raw_path.read_bytes()).hexdigest(),
        "external_processed_sha256": hashlib.sha256((root / "data/processed/Hanoi_test_final.csv").read_bytes()).hexdigest(),
        "run_timestamp": timestamp, "model_artifact": str(model_file.relative_to(root))})
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_experiment_review(root, report, evaluation)
    plot_experiment_results(root, report)
    return report


def write_experiment_review(root, report, evaluation):
    lines = ["# Kết quả thử nghiệm feature", "",
        f"Mã lượt chạy theo đồng hồ hệ thống: {report['run_timestamp']} (Asia/Ho_Chi_Minh).",
        f"Bộ ML chốt bằng CV: **{report['active_feature_set']}**, {len(report['selected_features'])} feature.",
        f"Model chốt bằng CV: **{report['selected_model_by_cv']}**.",
        "Các lựa chọn này được cố định trước khi mở test 2026; holdout/test chỉ dùng để đánh giá.", "",
        "## Phương pháp", "",
        "5 fold thời gian, gap bằng horizon 1 giờ. Mỗi fold dùng tối đa 4 năm train gần nhất; validation giữ đầy đủ.",
        "RF: 60 cây, depth 12; XGBoost: 200 cây, depth 5. Cấu hình cố định, chưa tuning lại.",
        "MAE/RMSE tính bằng °C. So sánh loại cột trên cùng giờ dự báo, bootstrap 2.000 lần theo ngày.",
        "Chỉ bỏ cột nếu cận trên suy giảm nằm trong 1% cho cả MAE/RMSE, cả RF/XGB, với hiệu chỉnh Bonferroni;",
        "đồng thời không có nhóm mùa/giờ tăng sai số quá 5%. Các cột được bỏ cùng nhau phải vượt qua kiểm tra chung.",
        "Bootstrap theo ngày chưa mô hình hóa tương quan qua nhiều ngày; CV phát triển không phải nested CV.", "",
        "## Sai số trên dữ liệu đánh giá", "",
        "| Tập | Bộ feature | Model | MAE | RMSE |", "|---|---|---|---:|---:|"]
    for row in evaluation.to_dict(orient="records"):
        lines.append(f"| {row['split']} | {row['feature_set']} | {row['Model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} |")
    lines.extend(["", "## Drop-column", "", "Delta dương nghĩa là bỏ biến làm tăng sai số.", "",
        "| Biến/nhóm bỏ | Model | ΔMAE | ΔRMSE | Cận trên MAE | Đủ điều kiện bỏ |",
        "|---|---|---:|---:|---:|---|"])
    for row in report["drop_column_scores"]:
        lines.append(f"| {row['feature_set']} | {row['Model']} | {row['delta_MAE']:.5f} | {row['delta_RMSE']:.5f} | {row['MAE_upper_bound']:.5f} | {'Có' if row['safe_to_remove'] else 'Không'} |")
    lines.extend(["", f"Các biến đã bỏ: {', '.join(report['removed_features']) or 'Không có; chưa đủ bằng chứng để bỏ an toàn.'}", "",
        "LSTM/GRU mặc định dùng weather + time trong chuỗi 24 giờ, giữ toàn bộ biến khí tượng;",
        "không dùng các lag/delta/rolling nhiệt độ thủ công. Kết luận ablation ML không được áp dụng tự động cho DL.",
        "Notebook DL có lựa chọn bộ input để chạy so sánh riêng; bảng trên là kết quả RF/XGBoost.", "",
        "## File kết quả", "",
        "- `feature_selection.json`: bộ chốt, cấu hình, confidence bounds, chi tiết theo mùa/giờ và SHA-256.",
        "- `../train_model/artifacts/feature_experiments/cv_summary.csv`: so sánh từng nhóm feature bằng CV.",
        "- `../train_model/artifacts/feature_experiments/drop_column_importance.csv`: kết quả loại cột.",
        "- `../train_model/artifacts/feature_experiments/metrics_by_season_hour.csv`: MAE/RMSE và số mẫu mỗi mùa/giờ.",
        "- `../train_model/artifacts/feature_experiments/evaluation_predictions.csv`: dự đoán ghép cặp để kiểm tra lại.",
        "- `../train_model/artifacts/feature_experiments/selected_models_refit_2026.joblib`: model được refit toàn bộ 2010–2025.",
        "", "Test 2026: DJF chỉ có tháng 1–2, SON chỉ có tháng 9; hai mùa này chưa đủ cả ba tháng.",
        "Test 2026 đã được đánh giá trong các lượt trước của dự án, vì vậy không còn là lockbox hoàn toàn mới.",
        "", "![So sánh feature](../train_model/artifacts/feature_experiments/feature_comparison.png)",
        "", "![Sai số theo mùa và giờ](../train_model/artifacts/feature_experiments/errors_by_season_hour.png)", ""])
    (root / "data_processing/feature_experiment_review.md").write_text("\n".join(lines), encoding="utf-8")


def plot_experiment_results(root, report):
    """Save standard plots for comparing feature groups and target-time errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(root) / "train_model/artifacts/feature_experiments"
    cv = pd.DataFrame(report["strategy_cv_scores"])
    evaluation = pd.DataFrame(report["holdout_scores"] + report["external_test_scores"])
    names = list(dict.fromkeys(["legacy_weather", "weather_cyclic_wind", "weather_time", "all_augmented",
                               report["active_feature_set"]]))
    labels = ["Weather (1h)", "Cyclic wind", "+ Time", "+ History"]
    if len(names) == 5:
        labels.append("Selected")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = ["#3478b6", "#df8125"]
    positions = np.arange(len(names))
    for ax, split in zip(axes, ["CV", "holdout", "test_2026"]):
        for i, model_name in enumerate(MODELS):
            if split == "CV":
                values = cv[(cv.Model == model_name)].set_index("feature_set").loc[names, "MAE_mean"]
                errors = cv[(cv.Model == model_name)].set_index("feature_set").loc[names, "MAE_std"]
            else:
                values = evaluation[(evaluation.Model == model_name) & (evaluation.split == split)].set_index("feature_set").loc[names, "MAE"]
                errors = None
            ax.bar(positions + (i - 0.5) * 0.36, values, width=0.36, yerr=errors,
                   capsize=3, color=colors[i], label=model_name)
        ax.set_xticks(positions, labels, rotation=30, ha="right")
        ax.set_title(split)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("MAE (Celsius)")
    axes[-1].legend()
    fig.suptitle("Feature comparison: explicit features at current hour; CV bars show fold standard deviation")
    fig.tight_layout()
    fig.savefig(output / "feature_comparison.png", dpi=160)
    plt.close(fig)

    slices = pd.DataFrame(report["evaluation_slices"])
    selected = slices[(slices.split == "test_2026") & (slices.feature_set == report["active_feature_set"])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, metric in enumerate(("MAE", "RMSE")):
        for i, model_name in enumerate(MODELS):
            part = selected[selected.Model == model_name]
            hourly = part[part.group == "hour"].copy()
            hourly["hour"] = hourly.value.astype(int)
            hourly = hourly.sort_values("hour")
            axes[j, 0].plot(hourly.hour, hourly[metric], marker="o", ms=3, color=colors[i], label=model_name)
            seasonal = part[part.group == "season"].set_index("value").reindex(["DJF", "MAM", "JJA", "SON"])
            axes[j, 1].bar(np.arange(4) + (i - 0.5) * 0.36, seasonal[metric], width=0.36,
                           color=colors[i], label=model_name)
        axes[j, 0].set_xlabel("Target hour (Hanoi time)")
        axes[j, 0].set_xticks(range(0, 24, 3))
        axes[j, 1].set_xticks(range(4), ["DJF (Jan-Feb)", "MAM", "JJA", "SON (Sep)"])
        for ax in axes[j]:
            ax.set_ylabel(f"{metric} (Celsius)")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
    axes[0, 1].legend()
    fig.suptitle(f"2026 evaluation by target hour and season: {report['active_feature_set']}")
    fig.tight_layout()
    fig.savefig(output / "errors_by_season_hour.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run_feature_experiments(Path(__file__).resolve().parents[1])
