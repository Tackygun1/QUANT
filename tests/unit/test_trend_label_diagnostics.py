from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant.research.trend_label_diagnostics import (
  TrendLabelDiagnosticsConfig,
  run_trend_label_diagnostics,
)


class TrendLabelDiagnosticsTests(unittest.TestCase):
  def test_writes_label_balance_and_candidate_threshold_summaries(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_dataset(root / "features")

      summary = run_trend_label_diagnostics(
        TrendLabelDiagnosticsConfig(
          input_root=root / "features",
          output_root=root / "reports",
          providers=("coinbase",),
          symbols=("BTC-USD",),
          horizons=(24,),
          label_columns=("future_trend_state_label_24_bar",),
          slope_dead_zone_bps_per_bar_candidates=(0.25, 0.5),
          slope_tstat_threshold_candidates=(0.5,),
        )
      )

      self.assertEqual(summary.series_processed, 1)
      payload = json.loads(summary.output_path.read_text(encoding="utf-8"))
      self.assertEqual(payload["rows"], 4)
      self.assertIn("future_trend_state_label_24_bar", payload["current_labels"])
      self.assertIn("24_bar", payload["candidate_state_labels"])
      self.assertEqual(len(payload["candidate_state_labels"]["24_bar"]), 2)

  def _write_dataset(self, input_root: Path) -> None:
    path = (
      input_root
      / "training_datasets"
      / "market_supervised"
      / "coinbase"
      / "BTC-USD"
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
          "is_closed",
          "future_trend_state_label_24_bar",
          "future_trend_slope_bps_per_bar_24_bar",
          "future_trend_slope_tstat_24_bar",
        ]
      )
      for index, label in enumerate((-1, 0, 1, 1)):
        writer.writerow(
          [
            "coinbase",
            "BTC-USD",
            "2026-04-12",
            f"2026-04-12T19:{index:02d}:00.000Z",
            "true",
            str(label),
            str(float(label)),
            "2.0",
          ]
        )


if __name__ == "__main__":
  unittest.main()
