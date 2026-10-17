"""Check the self-contained notebook without importing a training .py module."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "train_model/ml_models.ipynb"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
CODE_CELLS = ["".join(cell["source"]) for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]


def cell_containing(marker):
    return next(source for source in CODE_CELLS if marker in source)


def execute(source, namespace):
    with redirect_stdout(io.StringIO()):
        exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)


def helper_namespace():
    namespace = {
        "np": np, "pd": pd, "Path": Path, "TARGET": "temperature_2m",
        "VALIDATION_START": pd.Timestamp("2025-01-01"),
        "TEST_START": pd.Timestamp("2026-01-01"), "TEST_END": pd.Timestamp("2027-01-01"),
        "display": lambda *args: None,
    }
    execute(cell_containing("def load_observations("), namespace)
    execute(cell_containing("def make_sliding_windows("), namespace)
    return namespace


WEATHER_FEATURES = helper_namespace()["WEATHER_FEATURES"]


def observations(start, end):
    times = pd.date_range(start, end, freq="h")
    t = np.arange(len(times), dtype=float)
    frame = pd.DataFrame({"time": times})
    for i, name in enumerate(WEATHER_FEATURES):
        frame[name] = i + np.sin(t / (7 + i))
    frame["temperature_2m"] = 20 + 4 * np.sin(2 * np.pi * times.hour / 24) + np.sin(t / 100)
    frame["weather_code"] = 1.0
    return frame


class SlidingWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.ns = helper_namespace()

    def tearDown(self):
        self.temporary.cleanup()

    def weather(self, frame):
        path = self.root / "weather.csv"
        frame.to_csv(path, index=False)
        return self.ns["load_observations"](path)

    def windows(self, frame, features, window=24, horizon=1):
        return self.ns["make_sliding_windows"](frame, features, window, horizon)

    def selected_features(self, frame, horizon=1):
        self.ns.update(development=frame, HORIZON_HOURS=horizon)
        execute(cell_containing("first_validation_origin ="), self.ns)
        return self.ns["feature_columns"].copy(), self.ns["constant_columns"].copy()

    def test_window_values_labels_and_future_invariance(self):
        frame = self.weather(observations("2024-12-30", "2025-01-02"))
        features = ["temperature_2m", "relative_humidity_2m"]
        data = self.windows(frame, features, window=4, horizon=2)
        expected = frame[features].iloc[:4].to_numpy(dtype=np.float32)[::-1].reshape(-1)
        np.testing.assert_array_equal(data["X"][0], expected)
        self.assertEqual(data["y"][0], np.float32(frame.temperature_2m.iloc[5]))
        changed = frame.copy()
        changed.loc[changed.time > data["forecast_time"][10], features] += 1000
        rebuilt = self.windows(changed, features, window=4, horizon=2)
        np.testing.assert_array_equal(data["X"][:11], rebuilt["X"][:11])
        self.assertNotEqual(data["y"][10], rebuilt["y"][10])

    def test_calendar_split_uses_label_year_and_purges_long_horizon(self):
        frame = self.weather(observations("2024-12-28", "2025-12-31 23:00"))
        for horizon in (1, 6, 24):
            self.ns.update(windows=self.windows(frame, ["temperature_2m"], horizon=horizon),
                           WINDOW_HOURS=24, HORIZON_HOURS=horizon)
            execute(cell_containing("valid_start = int(label_times.searchsorted"), self.ns)
            train, valid = self.ns["train_label_times"], self.ns["valid_label_times"]
            self.assertTrue((train.year <= 2024).all())
            self.assertTrue((valid.year == 2025).all())
            self.assertEqual(valid[0], pd.Timestamp("2025-01-01"))
            self.assertLessEqual(train[-1], self.ns["valid_forecast_times"][0])
            self.assertLessEqual(valid[-1], pd.Timestamp("2026-01-01") - pd.Timedelta(hours=horizon))

    def test_constant_selection_never_uses_validation_values(self):
        frame = self.weather(observations("2024-12-28", "2025-01-03"))
        original = self.selected_features(frame)
        future = frame.time >= "2025-01-01"
        frame.loc[future, "weather_code"] = np.arange(future.sum())
        self.assertEqual(original, self.selected_features(frame))
        self.assertIn("weather_code", original[1])
        frame.loc[frame.time >= "2024-12-31 20:00", "weather_code"] = 99
        self.assertIn("weather_code", self.selected_features(frame, horizon=6)[1])

    def test_cleaning_is_forward_only_and_never_fills_target(self):
        raw = observations("2024-01-01", "2024-01-03")
        raw.loc[3:4, "relative_humidity_2m"] = np.nan
        frame = self.weather(raw)
        self.assertEqual(frame.relative_humidity_2m.iloc[3], raw.relative_humidity_2m.iloc[2])
        raw.loc[0, "relative_humidity_2m"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.weather(raw)
        raw.loc[0, "relative_humidity_2m"] = 50
        raw.loc[3, "temperature_2m"] = np.nan
        with self.assertRaisesRegex(ValueError, "nhãn giả"):
            self.weather(raw)

    def test_calendar_is_recomputed_and_future_columns_are_ignored(self):
        raw = observations("2024-02-28", "2024-03-01")
        raw["hour_sin"] = 10000
        raw["temperature_2m_future_1h"] = raw.temperature_2m.shift(-1)
        frame = self.weather(raw)
        self.assertNotIn("temperature_2m_future_1h", frame)
        self.assertAlmostEqual(frame.hour_sin.iloc[6], 1.0)
        leap = frame.loc[frame.time == "2024-02-29 00:00"].iloc[0]
        self.assertAlmostEqual(leap.day_of_year_sin, np.sin(2 * np.pi * 59 / 366))

    def test_missing_duplicate_hours_and_lfs_pointers_are_rejected(self):
        raw = observations("2024-01-01", "2024-01-03")
        for broken in (raw.drop(index=4), pd.concat([raw, raw.iloc[[4]]])):
            with self.assertRaises(ValueError):
                self.weather(broken)
        pointer = self.root / "pointer.csv"
        pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
        with self.assertRaisesRegex(ValueError, "Git LFS"):
            self.ns["load_observations"](pointer)

    def test_test_context_keeps_every_label_for_multiple_horizons(self):
        frame = self.weather(observations("2025-12-27", "2026-01-03 23:00"))
        development, test = frame[frame.time < "2026-01-01"], frame[frame.time >= "2026-01-01"]
        for horizon in (1, 6, 24):
            context = development.tail(24 + horizon - 1)
            combined = pd.concat([context, test], ignore_index=True)
            data = self.windows(combined, ["temperature_2m"], horizon=horizon)
            self.assertEqual(len(data["y"]), len(test))
            self.assertEqual(data["label_time"][0], pd.Timestamp("2026-01-01"))
            self.assertEqual(data["forecast_time"][0], pd.Timestamp("2026-01-01") - pd.Timedelta(hours=horizon))

    def test_notebook_pipeline_freezes_before_test_and_never_refits(self):
        data_dir = self.root / "data/processed"
        data_dir.mkdir(parents=True)
        raw = observations("2024-12-25", "2026-01-03 23:00")
        raw[raw.time < "2026-01-01"].to_csv(data_dir / "Hanoi_final.csv", index=False)
        test_path = data_dir / "Hanoi_test_final.csv"
        raw[raw.time >= "2026-01-01"].to_csv(test_path, index=False)
        fit_sizes, test_reads = [], []
        original_rf_fit, original_xgb_fit = RandomForestRegressor.fit, XGBRegressor.fit
        original_read_csv = pd.read_csv
        namespace = {"__name__": "__main__"}

        def tracked_rf(model, X, y, **kwargs):
            fit_sizes.append(len(y))
            return original_rf_fit(model, X, y, **kwargs)

        def tracked_xgb(model, X, y, **kwargs):
            fit_sizes.append(len(y))
            return original_xgb_fit(model, X, y, **kwargs)

        def tracked_read(path, *args, **kwargs):
            if Path(path).resolve() == test_path:
                self.assertTrue((namespace["RUN_DIR"] / "selection.json").exists())
                self.assertTrue((namespace["RUN_DIR"] / "best_model.joblib").exists())
                self.assertEqual(len(fit_sizes), 4)
                test_reads.append(str(path))
            return original_read_csv(path, *args, **kwargs)

        with patch.dict(os.environ, {"MPLCONFIGDIR": str(self.root / "mplcache"), "MPLBACKEND": "Agg"}), \
             patch.object(RandomForestRegressor, "fit", new=tracked_rf), \
             patch.object(XGBRegressor, "fit", new=tracked_xgb), \
             patch.object(pd, "read_csv", new=tracked_read):
            frozen_hash = None
            for source in CODE_CELLS:
                if "first_week = test_predictions" in source:
                    continue  # Plot layout is unchanged; exercise all data/model cells.
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        name = node.targets[0].id
                        if name == "PROJECT_ROOT" and isinstance(node.value, ast.Call):
                            node.value = ast.parse(f"Path({str(self.root)!r})", mode="eval").body
                        elif name in {"WINDOW_HOURS", "N_JOBS"}:
                            node.value = ast.Constant(3 if name == "WINDOW_HOURS" else 1)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"RandomForestRegressor", "XGBRegressor"}:
                        for keyword in node.keywords:
                            if keyword.arg in {"n_estimators", "max_depth"}:
                                keyword.value = ast.Constant(3 if keyword.arg == "n_estimators" else 2)
                ast.fix_missing_locations(tree)
                with redirect_stdout(io.StringIO()):
                    exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
                namespace["display"] = lambda *args: None
                if "joblib.dump(bundle, model_path" in source:
                    frozen_hash = namespace["file_sha256"](namespace["model_path"])
        run_dir = namespace["RUN_DIR"]
        self.assertEqual(run_dir.parent, self.root / "artifacts/ml_sliding_window")
        self.assertFalse((self.root / "train_model").exists())
        self.assertEqual(fit_sizes, [len(namespace["y_train"])] * 4)
        self.assertEqual(len(test_reads), 1)
        self.assertEqual(frozen_hash, namespace["file_sha256"](namespace["model_path"]))
        self.assertEqual(len(namespace["y_test"]), 72)
        self.assertEqual(namespace["test_label_times"][0], pd.Timestamp("2026-01-01"))
        saved = joblib.load(namespace["model_path"])
        self.assertEqual(saved["model_name"], namespace["ml_ranking"].iloc[0]["model"])
        with namespace["context_path"].open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "context đã bị thay đổi"):
            execute(cell_containing("frozen_selection = json.loads"), namespace)


if __name__ == "__main__":
    unittest.main()
