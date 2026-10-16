from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.research.trend_label_diagnostics import (
  TrendLabelDiagnosticsConfig,
  run_trend_label_diagnostics,
)
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  config = TrendLabelDiagnosticsConfig(
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "reports/trend_label_diagnostics")),
    dataset_name=str(config_value(args.dataset_name, payload, "dataset_name", "market_supervised")),
    providers=tuple(config_value(args.providers, payload, "providers", ())),
    symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
    horizons=tuple(config_value(args.horizons, payload, "horizons", (6, 24, 96))),
    label_columns=tuple(config_value(args.label_columns, payload, "label_columns", ())),
    slope_dead_zone_bps_per_bar_candidates=tuple(
      config_value(args.slope_dead_zone_bps_per_bar_candidates, payload, "slope_dead_zone_bps_per_bar_candidates", (0.25, 0.5, 1.0))
    ),
    slope_tstat_threshold_candidates=tuple(
      config_value(args.slope_tstat_threshold_candidates, payload, "slope_tstat_threshold_candidates", (0.5, 1.0, 1.5))
    ),
  )

  summary = run_trend_label_diagnostics(config)
  logging.getLogger(__name__).info(
    "Trend label diagnostics complete. series_processed=%s rows=%s output_path=%s",
    summary.series_processed,
    summary.rows,
    summary.output_path,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Inspect trend-label balance and threshold candidates.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--dataset-name")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--horizon", dest="horizons", type=int, action="append")
  parser.add_argument("--label-column", dest="label_columns", action="append")
  parser.add_argument("--slope-dead-zone-bps-per-bar", dest="slope_dead_zone_bps_per_bar_candidates", type=float, action="append")
  parser.add_argument("--slope-tstat-threshold", dest="slope_tstat_threshold_candidates", type=float, action="append")
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
