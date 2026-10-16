from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from quant.models.ensembles.stacked_meta import STACKED_META_MODEL_REGISTRY
from quant.models.ensembles.stacked_meta import HybridStackedDirectionMetaModel, StackedReturnMetaModel
from quant.training.train_meta import (
  MetaTrainingConfig,
  build_meta_sample_weights,
  resolve_stacked_prediction_runs,
  split_meta_training_datasets,
  train_meta_models,
)
from quant.training.datasets import TabularDataset


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "lightgbm is not installed")
class TrainMetaSmokeTests(unittest.TestCase):
  def test_trains_and_writes_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_stacked"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_csv(
        dataset_path,
        [
          "provider",
          "symbol",
          "feature_date",
          "candle_start",
          "candle_end",
          "is_closed",
          "close",
          "volume",
          "oof_ridge_return_prediction",
          "oof_histgb_return_prediction",
          "oof_transformer_return_forecaster_prediction",
          "direction_label_1_bar",
          "forward_return_1_bar",
        ],
        [
          [
            "binance",
            "BTCUSDT",
            "2026-04-12",
            f"2026-04-12T19:{minute:02d}:00.000Z",
            f"2026-04-12T19:{minute + 1:02d}:00.000Z",
            "true",
            str(100 + minute),
            str(10 + minute),
            str(0.0009 + minute * 0.00001),
            str(0.0010 + minute * 0.00001),
            str(0.0008 + minute * 0.00001),
            str(1 if minute % 3 == 0 else (-1 if minute % 3 == 1 else 0)),
            str(0.0012 * ((minute % 5) - 2)),
          ]
          for minute in range(24)
        ],
      )

      summary = train_meta_models(
        MetaTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          model_names=("lightgbm_return_meta",),
          minimum_rows=16,
          feature_groups=("price_core", "all_oof"),
          exclude_feature_columns=("volume",),
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      artifact_files = sorted((root / "models").rglob("*"))
      self.assertTrue(any(path.name == "model.pkl" for path in artifact_files))
      self.assertTrue(any(path.name == "metadata.json" for path in artifact_files))
      metadata_path = next(path for path in artifact_files if path.name == "metadata.json")
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      self.assertIn("oof_ridge_return_prediction", metadata["feature_columns"])
      self.assertNotIn("volume", metadata["feature_columns"])
      model_path = next(path for path in artifact_files if path.name == "model.pkl")
      loaded_model = StackedReturnMetaModel.load(model_path)
      predictions = loaded_model.predict([[110.0, 0.0011, 0.0012, 0.0010]])
      self.assertEqual(len(predictions), 1)

  def test_trains_hybrid_direction_meta_with_probability_calibration(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_stacked"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_csv(
        dataset_path,
        [
          "provider",
          "symbol",
          "feature_date",
          "candle_start",
          "candle_end",
          "is_closed",
          "close",
          "volume",
          "oof_ridge_return_prediction",
          "oof_histgb_return_prediction",
          "regime_hmm_state_index",
          "regime_bocpd_change_prob",
          "direction_label_1_bar",
        ],
        [
          [
            "binance",
            "BTCUSDT",
            "2026-04-12",
            f"2026-04-12T19:{minute:02d}:00.000Z",
            f"2026-04-12T19:{minute + 1:02d}:00.000Z",
            "true",
            str(100 + minute),
            str(10 + minute),
            str(0.0005 + minute * 0.00001),
            str(0.0007 + minute * 0.00001),
            str(minute % 3),
            "0.40" if minute % 7 == 0 else "0.10",
            str(1 if minute % 3 == 0 else (-1 if minute % 3 == 1 else 0)),
          ]
          for minute in range(90)
        ],
      )
      dataset_path.with_suffix(".manifest.json").write_text(
        json.dumps(
          {
            "prediction_runs": {
              "hybrid_transformer": "/tmp/models/walk_forward/hybrid_transformers/quantile_forecaster/binance/BTCUSDT/20260412T010000Z",
              "ridge_return": "/tmp/models/walk_forward/tabular/ridge_return/binance/BTCUSDT/20260412T010000Z",
            }
          },
          indent=2,
          sort_keys=True,
        ),
        encoding="utf-8",
      )

      summary = train_meta_models(
        MetaTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          model_names=("lightgbm_hybrid_direction_meta",),
          minimum_rows=48,
          probability_calibration_fraction=0.1,
          minimum_calibration_rows=8,
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      metadata_path = next(path for path in sorted((root / "models").rglob("metadata.json")))
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      calibration = metadata["extra"]["probability_calibration"]
      self.assertEqual(calibration["method"], "temperature")
      self.assertGreater(calibration["rows"], 0)
      self.assertEqual(
        metadata["extra"]["hybrid_transformer_artifact_dir"],
        "/tmp/models/walk_forward/hybrid_transformers/quantile_forecaster/binance/BTCUSDT/20260412T010000Z",
      )
      self.assertEqual(
        metadata["extra"]["stacked_prediction_runs"]["hybrid_transformer"],
        "/tmp/models/walk_forward/hybrid_transformers/quantile_forecaster/binance/BTCUSDT/20260412T010000Z",
      )
      model_path = metadata_path.parent / "model.pkl"
      loaded_model = HybridStackedDirectionMetaModel.load(model_path)
      probabilities = loaded_model.predict_proba([[140.0, 50.0, 0.0011, 0.0013, 2.0, 0.1]])
      self.assertEqual(len(probabilities), 1)
      self.assertAlmostEqual(sum(probabilities[0]), 1.0, places=6)

  def test_trains_hybrid_direction_meta_with_extra_regularization_controls(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_stacked"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_csv(
        dataset_path,
        [
          "provider",
          "symbol",
          "feature_date",
          "candle_start",
          "candle_end",
          "is_closed",
          "close",
          "volume",
          "oof_ridge_return_prediction",
          "oof_histgb_return_prediction",
          "trend_return_24_bar",
          "structure_supply_zone_distance_24_bar",
          "structure_demand_zone_distance_24_bar",
          "direction_label_1_bar",
        ],
        [
          [
            "binance",
            "BTCUSDT",
            "2026-04-12",
            f"2026-04-12T19:{minute:02d}:00.000Z",
            f"2026-04-12T19:{minute + 1:02d}:00.000Z",
            "true",
            str(100 + minute),
            str(10 + minute),
            str(0.0005 + minute * 0.00001),
            str(0.0007 + minute * 0.00001),
            str(0.002 if minute % 3 == 0 else (-0.002 if minute % 3 == 1 else 0.0)),
            str(0.001 + minute * 0.00002),
            str(-0.001 + minute * 0.00001),
            str(1 if minute % 3 == 0 else (-1 if minute % 3 == 1 else 0)),
          ]
          for minute in range(96)
        ],
      )

      summary = train_meta_models(
        MetaTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          model_names=("lightgbm_hybrid_direction_meta",),
          minimum_rows=48,
          probability_calibration_method="",
          model_params={
            "lightgbm_hybrid_direction_meta": {
              "learning_rate": 0.05,
              "n_estimators": 100,
              "num_leaves": 15,
              "max_depth": 3,
              "min_child_samples": 24,
              "min_split_gain": 0.05,
              "feature_fraction_bynode": 0.7,
              "path_smooth": 10.0,
            }
          },
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      metadata_path = next(path for path in sorted((root / "models").rglob("metadata.json")))
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      params = metadata["extra"]["model_params"]
      self.assertEqual(params["max_depth"], 3)
      self.assertAlmostEqual(params["min_split_gain"], 0.05)
      self.assertAlmostEqual(params["feature_fraction_bynode"], 0.7)
      self.assertAlmostEqual(params["path_smooth"], 10.0)

  def test_resolves_stacked_prediction_runs_from_manifests(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      dataset_root = (
        root
        / "features"
        / "training_datasets"
        / "market_hybrid_stacked"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
      )
      dataset_root.mkdir(parents=True, exist_ok=True)
      (dataset_root / "12.manifest.json").write_text(
        json.dumps(
          {
            "prediction_runs": {
              "hybrid_transformer": "/tmp/run_a",
              "ridge_return": "/tmp/ridge_a",
            }
          }
        ),
        encoding="utf-8",
      )
      (dataset_root / "13.manifest.json").write_text(
        json.dumps(
          {
            "prediction_runs": {
              "hybrid_transformer": "/tmp/run_a",
              "ridge_return": "/tmp/ridge_b",
            }
          }
        ),
        encoding="utf-8",
      )
      (dataset_root / "14.manifest.json").write_text(
        json.dumps(
          {
            "prediction_runs": {
              "hybrid_transformer": "/tmp/run_b",
              "ridge_return": "/tmp/ridge_b",
            }
          }
        ),
        encoding="utf-8",
      )

      runs = resolve_stacked_prediction_runs(
        root / "features",
        dataset_name="market_hybrid_stacked",
        provider="binance",
        symbol="BTCUSDT",
      )

      self.assertEqual(runs["hybrid_transformer"], "/tmp/run_a")
      self.assertEqual(runs["ridge_return"], "/tmp/ridge_b")

  def test_hybrid_direction_builder_preserves_regularization_and_passthrough_params(self) -> None:
    model = STACKED_META_MODEL_REGISTRY["lightgbm_hybrid_direction_meta"]["builder"](
      {
        "learning_rate": 0.04,
        "n_estimators": 120,
        "num_leaves": 15,
        "max_depth": 3,
        "min_child_samples": 32,
        "min_split_gain": 0.1,
        "feature_fraction_bynode": 0.6,
        "path_smooth": 8.0,
        "interaction_constraints": [[0, 1, 2], [3, 4]],
      }
    )
    self.assertEqual(model.max_depth, 3)
    self.assertAlmostEqual(model.min_split_gain, 0.1)
    self.assertAlmostEqual(model.feature_fraction_bynode, 0.6)
    self.assertAlmostEqual(model.path_smooth, 8.0)
    self.assertEqual(model.extra_model_params["interaction_constraints"], [[0, 1, 2], [3, 4]])

  def test_builds_trend_balanced_sample_weights(self) -> None:
    rows = [
      {"trend_return_24_bar": "0.004"},
      {"trend_return_24_bar": "0.003"},
      {"trend_return_24_bar": "-0.003"},
      {"trend_return_24_bar": "0.0001"},
    ]
    weights = build_meta_sample_weights(
      rows=rows,
      targets=[0.001, 0.002, -0.001, 0.0],
      scheme="trend_regime_balance",
      trend_column="trend_return_24_bar",
      trend_threshold_bps=20.0,
      regime_column="regime_hmm_state_index",
      regime_change_column="regime_bocpd_change_prob",
      regime_change_threshold=0.35,
      regime_change_boost=1.0,
      floor=0.5,
      cap=3.0,
    )
    self.assertEqual(len(weights), 4)
    self.assertAlmostEqual(weights[0], 0.6666666666666666)
    self.assertAlmostEqual(weights[1], 0.6666666666666666)
    self.assertAlmostEqual(weights[2], 1.3333333333333333)
    self.assertAlmostEqual(weights[3], 1.3333333333333333)

  def test_builds_trend_target_balanced_sample_weights(self) -> None:
    rows = [
      {"trend_return_24_bar": "0.004"},
      {"trend_return_24_bar": "0.004"},
      {"trend_return_24_bar": "0.004"},
      {"trend_return_24_bar": "-0.004"},
      {"trend_return_24_bar": "-0.004"},
    ]
    weights = build_meta_sample_weights(
      rows=rows,
      targets=[0.001, 0.002, -0.001, -0.002, -0.003],
      scheme="trend_regime_target_balance",
      trend_column="trend_return_24_bar",
      trend_threshold_bps=20.0,
      regime_column="regime_hmm_state_index",
      regime_change_column="regime_bocpd_change_prob",
      regime_change_threshold=0.35,
      regime_change_boost=1.0,
      floor=0.5,
      cap=3.0,
    )
    self.assertEqual(len(weights), 5)
    self.assertGreater(weights[2], weights[0])
    self.assertGreater(weights[2], weights[1])
    self.assertGreater(weights[2], weights[3])
    self.assertGreater(weights[2], weights[4])

  def test_builds_hmm_regime_target_weights_with_transition_boost(self) -> None:
    rows = [
      {"regime_hmm_state_index": "2", "regime_bocpd_change_prob": "0.10"},
      {"regime_hmm_state_index": "2", "regime_bocpd_change_prob": "0.60"},
      {"regime_hmm_state_index": "0", "regime_bocpd_change_prob": "0.20"},
      {"regime_hmm_state_index": "1", "regime_bocpd_change_prob": "0.55"},
    ]
    weights = build_meta_sample_weights(
      rows=rows,
      targets=[1, 1, -1, 0],
      scheme="hmm_regime_target_balance",
      trend_column="trend_return_24_bar",
      trend_threshold_bps=20.0,
      regime_column="regime_hmm_state_index",
      regime_change_column="regime_bocpd_change_prob",
      regime_change_threshold=0.5,
      regime_change_boost=1.5,
      floor=0.5,
      cap=3.0,
    )
    self.assertEqual(len(weights), 4)
    self.assertGreater(weights[1], weights[0])
    self.assertGreater(weights[3], weights[2])

  def test_split_meta_training_datasets_reserves_tail_for_calibration(self) -> None:
    dataset = TabularDataset(
      feature_names=["close"],
      target_column="direction_label_1_bar",
      rows=[{"candle_start": f"2026-04-12T19:{minute:02d}:00.000Z"} for minute in range(40)],
      features=[[float(minute)] for minute in range(40)],
      targets=[-1, 0, 1, 0] * 10,
    )

    fit_dataset, calibration_dataset = split_meta_training_datasets(
      model_name="lightgbm_hybrid_direction_meta",
      task_type="classification",
      target_column="direction_label_1_bar",
      train_dataset=dataset,
      calibration_method="temperature",
      calibration_fraction=0.2,
      minimum_calibration_rows=8,
      calibration_model_names=("lightgbm_hybrid_direction_meta",),
    )

    self.assertIsNotNone(calibration_dataset)
    assert calibration_dataset is not None
    self.assertEqual(len(fit_dataset.rows), 32)
    self.assertEqual(len(calibration_dataset.rows), 8)
    self.assertEqual(fit_dataset.rows[-1]["candle_start"], "2026-04-12T19:31:00.000Z")
    self.assertEqual(calibration_dataset.rows[0]["candle_start"], "2026-04-12T19:32:00.000Z")

  @staticmethod
  def _write_csv(path: Path, fieldnames: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(fieldnames)
      writer.writerows(rows)


if __name__ == "__main__":
  unittest.main()
