from __future__ import annotations

import argparse
import logging
from decimal import Decimal
from pathlib import Path

from quant.labels.market_labels import LabelBuildConfig, build_market_labels


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)

  config = LabelBuildConfig(
    input_root=args.input_root,
    output_root=args.output_root,
    providers=tuple(args.providers or ()),
    symbols=tuple(args.symbols or ()),
    horizons=tuple(args.horizons or (1, 6, 24, 96)),
    direction_dead_zone_bps=Decimal(str(args.direction_dead_zone_bps)),
    trade_cost_bps=Decimal(str(args.trade_cost_bps)),
    realized_volatility_window=args.realized_volatility_window,
    trend_slope_dead_zone_bps_per_bar=Decimal(str(args.trend_slope_dead_zone_bps_per_bar)),
    trend_slope_tstat_threshold=args.trend_slope_tstat_threshold,
    trend_efficiency_threshold=args.trend_efficiency_threshold,
    triple_barrier_profit_bps=Decimal(str(args.triple_barrier_profit_bps)),
    triple_barrier_stop_bps=Decimal(str(args.triple_barrier_stop_bps)),
    limit_series=args.limit_series,
  )

  summary = build_market_labels(config)
  logging.getLogger(__name__).info(
    "Label build complete. series_processed=%s output_files_written=%s label_rows_written=%s",
    summary.series_processed,
    summary.output_files_written,
    summary.label_rows_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Build supervised market labels from candle-anchored feature tables.",
  )
  parser.add_argument(
    "--input-root",
    type=Path,
    default=Path("data/features"),
    help="Root directory containing feature tables.",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("data/features"),
    help="Root directory for label tables.",
  )
  parser.add_argument(
    "--provider",
    dest="providers",
    action="append",
    help="Optional provider filter. Repeat for multiple values.",
  )
  parser.add_argument(
    "--symbol",
    dest="symbols",
    action="append",
    help="Optional symbol filter. Repeat for multiple values.",
  )
  parser.add_argument(
    "--horizon",
    dest="horizons",
    type=int,
    action="append",
    help="Forward return horizon in bars. Repeat for multiple horizons.",
  )
  parser.add_argument(
    "--direction-dead-zone-bps",
    type=float,
    default=5.0,
    help="Dead-zone threshold for direction labels in basis points.",
  )
  parser.add_argument(
    "--trade-cost-bps",
    type=float,
    default=10.0,
    help="Cost threshold for tradeability labels in basis points.",
  )
  parser.add_argument(
    "--realized-volatility-window",
    type=int,
    default=5,
    help="Number of future one-bar returns to use for realized volatility labels.",
  )
  parser.add_argument(
    "--trend-slope-dead-zone-bps-per-bar",
    type=float,
    default=0.5,
    help="Minimum future trend slope in bps per bar before labeling up/down trend.",
  )
  parser.add_argument(
    "--trend-slope-tstat-threshold",
    type=float,
    default=1.0,
    help="Minimum absolute future trend slope t-stat before labeling a trend.",
  )
  parser.add_argument(
    "--trend-efficiency-threshold",
    type=float,
    default=0.35,
    help="Minimum path efficiency for future trend persistence labels.",
  )
  parser.add_argument(
    "--triple-barrier-profit-bps",
    type=float,
    default=20.0,
    help="Upper barrier in basis points for triple-barrier direction labels.",
  )
  parser.add_argument(
    "--triple-barrier-stop-bps",
    type=float,
    default=20.0,
    help="Lower barrier in basis points for triple-barrier direction labels.",
  )
  parser.add_argument(
    "--limit-series",
    type=int,
    help="Optional cap on provider/symbol series to process.",
  )
  parser.add_argument(
    "--log-level",
    default="INFO",
    help="Logging level. Defaults to INFO.",
  )
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
