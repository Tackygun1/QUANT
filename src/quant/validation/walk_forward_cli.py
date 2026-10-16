from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from quant.models.transformers.return_forecaster import TransformerForecasterConfig
from quant.utils.config import config_value, load_json_config
from quant.validation.walk_forward import (
  TabularWalkForwardConfig,
  TransformerWalkForwardConfig,
  evaluate_tabular_walk_forward,
  evaluate_transformer_walk_forward,
)


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  family = config_value(args.family, payload, "family", "tabular")

  if family == "tabular":
    config = TabularWalkForwardConfig(
      input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
      output_root=Path(config_value(args.output_root, payload, "output_root", "data/models")),
      providers=tuple(config_value(args.providers, payload, "providers", ())),
      symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
      model_names=tuple(config_value(args.models, payload, "model_names", ("ridge_return", "logistic_direction", "histgb_return", "histgb_direction"))),
      initial_train_rows=int(config_value(args.initial_train_rows, payload, "initial_train_rows", 256)),
      validation_rows=int(config_value(args.validation_rows, payload, "validation_rows", 64)),
      step_rows=_optional_int(args.step_rows if args.step_rows is not None else payload.get("step_rows")),
      embargo_rows=int(config_value(args.embargo_rows, payload, "embargo_rows", 0)),
      purge_rows=_optional_int(config_value(args.purge_rows, payload, "purge_rows")),
      minimum_rows=int(config_value(args.minimum_rows, payload, "minimum_rows", 320)),
      target_overrides=payload.get("target_overrides"),
      model_params=payload.get("model_params"),
      config_path=args.config,
    )
    summary = evaluate_tabular_walk_forward(config)
  elif family == "transformer":
    transformer_payload = payload.get("transformer", {})
    config = TransformerWalkForwardConfig(
      input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
      output_root=Path(config_value(args.output_root, payload, "output_root", "data/models")),
      providers=tuple(config_value(args.providers, payload, "providers", ())),
      symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
      target_column=config_value(args.target_column, payload, "target_column", "forward_return_1_bar"),
      initial_train_rows=int(config_value(args.initial_train_rows, payload, "initial_train_rows", 256)),
      validation_rows=int(config_value(args.validation_rows, payload, "validation_rows", 64)),
      step_rows=_optional_int(args.step_rows if args.step_rows is not None else payload.get("step_rows")),
      embargo_rows=int(config_value(args.embargo_rows, payload, "embargo_rows", 0)),
      purge_rows=_optional_int(config_value(args.purge_rows, payload, "purge_rows")),
      minimum_rows=int(config_value(args.minimum_rows, payload, "minimum_rows", 320)),
      config_path=args.config,
      transformer=TransformerForecasterConfig(
        lookback=int(config_value(args.lookback, transformer_payload, "lookback", 32)),
        d_model=int(config_value(args.d_model, transformer_payload, "d_model", 64)),
        nhead=int(config_value(args.nhead, transformer_payload, "nhead", 4)),
        num_layers=int(config_value(args.num_layers, transformer_payload, "num_layers", 2)),
        dim_feedforward=int(config_value(args.dim_feedforward, transformer_payload, "dim_feedforward", 128)),
        dropout=float(config_value(args.dropout, transformer_payload, "dropout", 0.1)),
        batch_size=int(config_value(args.batch_size, transformer_payload, "batch_size", 32)),
        epochs=int(config_value(args.epochs, transformer_payload, "epochs", 20)),
        learning_rate=float(config_value(args.learning_rate, transformer_payload, "learning_rate", 1e-3)),
        weight_decay=float(config_value(args.weight_decay, transformer_payload, "weight_decay", 1e-4)),
        validation_fraction=float(config_value(None, transformer_payload, "validation_fraction", 0.25)),
        device=config_value(args.device, transformer_payload, "device", "cpu"),
      ),
    )
    summary = evaluate_transformer_walk_forward(config)
  else:
    raise ValueError(f"Unsupported family: {family}")

  logging.getLogger(__name__).info(
    "Walk-forward run complete. family=%s runs_attempted=%s runs_completed=%s folds_completed=%s artifacts_written=%s",
    family,
    summary.runs_attempted,
    summary.runs_completed,
    summary.folds_completed,
    summary.artifacts_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run expanding-window walk-forward validation on the joined supervised dataset.")
  parser.add_argument("--family", choices=("tabular", "transformer"))
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--model", dest="models", action="append")
  parser.add_argument("--target-column")
  parser.add_argument("--initial-train-rows", type=int)
  parser.add_argument("--validation-rows", type=int)
  parser.add_argument("--step-rows", type=int)
  parser.add_argument("--embargo-rows", type=int)
  parser.add_argument("--purge-rows", type=int)
  parser.add_argument("--minimum-rows", type=int)
  parser.add_argument("--lookback", type=int)
  parser.add_argument("--d-model", type=int)
  parser.add_argument("--nhead", type=int)
  parser.add_argument("--num-layers", type=int)
  parser.add_argument("--dim-feedforward", type=int)
  parser.add_argument("--dropout", type=float)
  parser.add_argument("--batch-size", type=int)
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--learning-rate", type=float)
  parser.add_argument("--weight-decay", type=float)
  parser.add_argument("--device")
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


def _optional_int(value: Any) -> int | None:
  if value in (None, "", "none"):
    return None
  return int(value)
