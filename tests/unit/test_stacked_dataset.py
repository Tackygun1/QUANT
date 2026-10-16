from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant.training.stacked_dataset import PredictionSource, StackedDatasetConfig, build_stacked_datasets


class StackedDatasetTests(unittest.TestCase):
  def test_builds_stacked_dataset_from_walk_forward_predictions(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_supervised_dataset(root / "features")
      self._write_prediction_run(
        root / "models" / "walk_forward" / "tabular" / "ridge_return" / "binance" / "BTCUSDT" / "20260412T010000Z",
        predictions={
          "2026-04-12T19:01:00.000Z": "0.001",
          "2026-04-12T19:02:00.000Z": "0.002",
        },
      )
      self._write_prediction_run(
        root / "models" / "walk_forward" / "transformers" / "return_forecaster" / "binance" / "BTCUSDT" / "20260412T010000Z",
        predictions={
          "2026-04-12T19:02:00.000Z": "0.003",
        },
      )

      summary = build_stacked_datasets(
        StackedDatasetConfig(
          input_root=root / "features",
          prediction_root=root / "models" / "walk_forward",
          output_root=root / "features",
          providers=("binance",),
          symbols=("BTCUSDT",),
          prediction_sources=(
            PredictionSource("tabular", "ridge_return", "ridge_return"),
            PredictionSource("transformers", "return_forecaster", "transformer_return_forecaster"),
          ),
        )
      )

      self.assertEqual(summary.datasets_processed, 1)
      output_path = (
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
      with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]["candle_start"], "2026-04-12T19:02:00.000Z")
      self.assertEqual(rows[0]["oof_ridge_return_prediction"], "0.002")
      self.assertEqual(rows[0]["oof_transformer_return_forecaster_prediction"], "0.003")

      manifest_path = output_path.with_suffix(".manifest.json")
      payload = json.loads(manifest_path.read_text(encoding="utf-8"))
      self.assertIn("ridge_return", payload["prediction_runs"])

  def test_builds_stacked_dataset_from_multi_column_predictions(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_supervised_dataset(root / "features")
      self._write_prediction_run(
        root / "models" / "walk_forward" / "tabular" / "histgb_return" / "binance" / "BTCUSDT" / "20260412T010000Z",
        predictions={
          "2026-04-12T19:02:00.000Z": {"prediction": "0.002"},
        },
      )
      self._write_prediction_run(
        root / "models" / "walk_forward" / "hybrid_transformers" / "quantile_forecaster" / "binance" / "BTCUSDT" / "20260412T010000Z",
        predictions={
          "2026-04-12T19:02:00.000Z": {
            "prediction_return_q10_1_bar": "-0.001",
            "prediction_return_q50_1_bar": "0.0025",
            "prediction_return_q90_1_bar": "0.004",
            "prediction_return_spread_1_bar": "0.005",
          },
        },
      )

      summary = build_stacked_datasets(
        StackedDatasetConfig(
          input_root=root / "features",
          prediction_root=root / "models" / "walk_forward",
          output_root=root / "features",
          providers=("binance",),
          symbols=("BTCUSDT",),
          prediction_sources=(
            PredictionSource("tabular", "histgb_return", "histgb_return"),
            PredictionSource(
              "hybrid_transformers",
              "quantile_forecaster",
              "hybrid_transformer",
              source_columns=(
                "prediction_return_q10_1_bar",
                "prediction_return_q50_1_bar",
                "prediction_return_q90_1_bar",
                "prediction_return_spread_1_bar",
              ),
            ),
          ),
        )
      )

      self.assertEqual(summary.datasets_processed, 1)
      output_path = (
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
      with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]["oof_hybrid_transformer_return_q50_1_bar"], "0.0025")
      self.assertEqual(rows[0]["oof_hybrid_transformer_return_spread_1_bar"], "0.005")

  @staticmethod
  def _write_supervised_dataset(input_root: Path) -> None:
    path = (
      input_root
      / "training_datasets"
      / "market_supervised"
      / "binance"
      / "BTCUSDT"
      / "2026"
      / "04"
      / "12.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(
        [
          "provider",
          "symbol",
          "feature_date",
          "candle_start",
          "candle_end",
          "is_closed",
          "close",
          "volume",
          "forward_return_1_bar",
          "direction_label_1_bar",
        ]
      )
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "2026-04-12T19:01:00.000Z", "2026-04-12T19:02:00.000Z", "true", "100", "10", "0.001", "1"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "2026-04-12T19:02:00.000Z", "2026-04-12T19:03:00.000Z", "true", "101", "11", "0.002", "1"])

  @staticmethod
  def _write_prediction_run(run_root: Path, *, predictions: dict[str, str | dict[str, str]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "task_type", "fold_index", "provider", "symbol", "candle_start"]
    extra_fields: list[str] = []
    for payload in predictions.values():
      if isinstance(payload, dict):
        for field in payload.keys():
          if field not in extra_fields:
            extra_fields.append(field)
      else:
        if "target" not in extra_fields:
          extra_fields.append("target")
        if "prediction" not in extra_fields:
          extra_fields.append("prediction")
    with (run_root / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow([*fieldnames, *extra_fields])
      for candle_start, payload in predictions.items():
        row_payload = {
          "model_name": "model",
          "task_type": "regression",
          "fold_index": "1",
          "provider": "binance",
          "symbol": "BTCUSDT",
          "candle_start": candle_start,
        }
        if isinstance(payload, dict):
          row_payload.update(payload)
        else:
          row_payload["target"] = "0.0"
          row_payload["prediction"] = payload
        writer.writerow([row_payload.get(field, "") for field in [*fieldnames, *extra_fields]])


if __name__ == "__main__":
  unittest.main()
