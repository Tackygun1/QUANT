from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.models.transformers.return_forecaster import TransformerForecasterConfig
from quant.training.train_transformer import TransformerTrainingConfig, train_transformer_forecasters
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  transformer_payload = payload.get("transformer", {})
  transformer_config = TransformerForecasterConfig(
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
    validation_fraction=float(config_value(args.validation_fraction, transformer_payload, "validation_fraction", 0.25)),
    device=config_value(args.device, transformer_payload, "device", "cpu"),
  )

  config = TransformerTrainingConfig(
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "data/models")),
    providers=tuple(config_value(args.providers, payload, "providers", ())),
    symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
    target_column=config_value(args.target_column, payload, "target_column", "forward_return_1_bar"),
    minimum_rows=int(config_value(args.minimum_rows, payload, "minimum_rows", 64)),
    config_path=args.config,
    transformer=transformer_config,
  )
  summary = train_transformer_forecasters(config)
  logging.getLogger(__name__).info(
    "Transformer training complete. runs_attempted=%s runs_completed=%s artifacts_written=%s",
    summary.runs_attempted,
    summary.runs_completed,
    summary.artifacts_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Train a Transformer return forecaster on the joined supervised dataset.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--target-column")
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
  parser.add_argument("--validation-fraction", type=float)
  parser.add_argument("--device")
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
