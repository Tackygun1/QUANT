from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.training.stacked_dataset import (
  PredictionSource,
  StackedDatasetConfig,
  build_stacked_datasets,
)
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  source_payload = payload.get("prediction_sources")
  prediction_sources = tuple(
    PredictionSource(
      family=item["family"],
      model_name=item["model_name"],
      alias=item.get("alias", item["model_name"]),
      source_columns=tuple(item.get("source_columns", ("prediction",))),
      output_columns=tuple(item.get("output_columns", ())),
    )
    for item in (source_payload or _default_prediction_sources())
  )

  config = StackedDatasetConfig(
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    prediction_root=Path(config_value(args.prediction_root, payload, "prediction_root", "data/models/walk_forward")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "data/features")),
    input_dataset_name=config_value(args.input_dataset_name, payload, "input_dataset_name", "market_supervised"),
    output_dataset_name=config_value(args.output_dataset_name, payload, "output_dataset_name", "market_stacked"),
    providers=tuple(config_value(args.providers, payload, "providers", ())),
    symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
    limit_datasets=config_value(args.limit_datasets, payload, "limit_datasets"),
    prediction_sources=prediction_sources,
  )

  summary = build_stacked_datasets(config)
  logging.getLogger(__name__).info(
    "Stacked dataset build complete. datasets_processed=%s output_files_written=%s stacked_rows_written=%s prediction_runs_used=%s",
    summary.datasets_processed,
    summary.output_files_written,
    summary.stacked_rows_written,
    summary.prediction_runs_used,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Join walk-forward predictions onto the supervised dataset to build a stacked training table.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--prediction-root")
  parser.add_argument("--output-root")
  parser.add_argument("--input-dataset-name")
  parser.add_argument("--output-dataset-name")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--limit-datasets", type=int)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


def _default_prediction_sources() -> list[dict[str, str]]:
  return [
    {"family": "tabular", "model_name": "ridge_return", "alias": "ridge_return"},
    {"family": "tabular", "model_name": "histgb_return", "alias": "histgb_return"},
    {"family": "tabular", "model_name": "logistic_direction", "alias": "logistic_direction"},
    {"family": "tabular", "model_name": "histgb_direction", "alias": "histgb_direction"},
    {"family": "transformers", "model_name": "return_forecaster", "alias": "transformer_return_forecaster"},
  ]
