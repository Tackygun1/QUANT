from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant.research.trend_recognition_comparison import (
  TrendRecognitionComparisonConfig,
  run_trend_recognition_comparison,
)


class TrendRecognitionComparisonTests(unittest.TestCase):
  def test_compares_predictions_and_ranks_gate_profiles(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_features(root / "features")
      baseline_path = root / "baseline.csv"
      trend_path = root / "trend.csv"
      self._write_predictions(baseline_path, trend_path)

      summary = run_trend_recognition_comparison(
        TrendRecognitionComparisonConfig(
          feature_root=root / "features",
          baseline_prediction_path=baseline_path,
          trend_prediction_path=trend_path,
          output_root=root / "reports",
          gate_thresholds=(0.55,),
          chop_thresholds=(0.6,),
          countertrend_scales=(0.0, 0.5),
          calibration_fraction=0.5,
          minimum_calibration_rows=1,
        )
      )

      self.assertEqual(summary.common_rows, 4)
      self.assertEqual(summary.calibration_rows, 2)
      self.assertEqual(summary.holdout_rows, 2)
      self.assertIsNotNone(summary.best_profile)
      self.assertIsNotNone(summary.calibrated_holdout_profile)
      self.assertTrue(summary.output_path.exists())
      payload = json.loads(summary.output_path.read_text(encoding="utf-8"))
      self.assertIn("trend_probability_calibration_common", payload)
      self.assertIn("selected_holdout_profile", payload["calibration"])
      self.assertIn("recommended_sizing", payload["calibration"]["selected_holdout_profile"])

  def _write_features(self, feature_root: Path) -> None:
    path = feature_root / "training_datasets" / "market_supervised" / "coinbase" / "BTC-USD" / "2026" / "04" / "12.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(
        handle,
        fieldnames=[
          "provider",
          "symbol",
          "candle_start",
          "future_trend_state_label_24_bar",
          "regime_bocpd_change_prob",
        ],
      )
      writer.writeheader()
      for index, target in enumerate((1, 0, -1, 1)):
        writer.writerow(
          {
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "candle_start": f"2026-04-12T00:0{index}:00Z",
            "future_trend_state_label_24_bar": target,
            "regime_bocpd_change_prob": "0.5" if index in {1, 2} else "0.1",
          }
        )

  def _write_predictions(self, baseline_path: Path, trend_path: Path) -> None:
    baseline_rows = [
      ("2026-04-12T00:00:00Z", "0.01"),
      ("2026-04-12T00:01:00Z", "0.01"),
      ("2026-04-12T00:02:00Z", "-0.01"),
      ("2026-04-12T00:03:00Z", "-0.01"),
    ]
    trend_rows = [
      ("2026-04-12T00:00:00Z", "1", "0.80", "0.10", "0.10", "0.01"),
      ("2026-04-12T00:01:00Z", "0", "0.20", "0.10", "0.70", "0.01"),
      ("2026-04-12T00:02:00Z", "-1", "0.10", "0.80", "0.10", "-0.01"),
      ("2026-04-12T00:03:00Z", "1", "0.70", "0.10", "0.20", "0.01"),
    ]
    with baseline_path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=["candle_start", "prediction_return_q50_24_bar"])
      writer.writeheader()
      for candle_start, prediction in baseline_rows:
        writer.writerow({"candle_start": candle_start, "prediction_return_q50_24_bar": prediction})
    with trend_path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(
        handle,
        fieldnames=[
          "candle_start",
          "prediction_trend_state_24_bar",
          "prediction_probability_uptrend_24_bar",
          "prediction_probability_downtrend_24_bar",
          "prediction_probability_chop_24_bar",
          "prediction_return_q50_24_bar",
        ],
      )
      writer.writeheader()
      for row in trend_rows:
        writer.writerow(
          {
            "candle_start": row[0],
            "prediction_trend_state_24_bar": row[1],
            "prediction_probability_uptrend_24_bar": row[2],
            "prediction_probability_downtrend_24_bar": row[3],
            "prediction_probability_chop_24_bar": row[4],
            "prediction_return_q50_24_bar": row[5],
          }
        )


if __name__ == "__main__":
  unittest.main()
