from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.training.train_tabular import TabularTrainingConfig, train_tabular_models
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  model_names = tuple(config_value(args.models, payload, "model_names", ()))
  if not model_names:
    model_names = ("ridge_return", "logistic_direction", "histgb_return", "histgb_direction")

  config = TabularTrainingConfig(
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "data/models")),
    providers=tuple(config_value(args.providers, payload, "providers", ())),
    symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
    model_names=model_names,
    validation_fraction=float(config_value(args.validation_fraction, payload, "validation_fraction", 0.34)),
    minimum_rows=int(config_value(args.minimum_rows, payload, "minimum_rows", 10)),
    target_overrides=payload.get("target_overrides"),
    model_params=payload.get("model_params"),
    config_path=args.config,
  )
  summary = train_tabular_models(config)
  logging.getLogger(__name__).info(
    "Tabular training complete. runs_attempted=%s runs_completed=%s artifacts_written=%s",
    summary.runs_attempted,
    summary.runs_completed,
    summary.artifacts_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Train baseline tabular models on the joined supervised dataset.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--model", dest="models", action="append")
  parser.add_argument("--validation-fraction", type=float)
  parser.add_argument("--minimum-rows", type=int)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
