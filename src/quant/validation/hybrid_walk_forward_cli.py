from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from quant.models.transformers.hybrid_forecaster import HybridTransformerConfig, build_target_columns
from quant.utils.config import config_value, load_json_config
from quant.validation.hybrid_walk_forward import (
  HybridTransformerWalkForwardConfig,
  evaluate_hybrid_transformer_walk_forward,
)


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  transformer_payload = payload.get("transformer", {})
  horizons = tuple(config_value(args.horizons, transformer_payload, "horizons", (1, 6, 24, 96)))
  quantiles = tuple(config_value(args.quantiles, transformer_payload, "quantiles", (0.1, 0.5, 0.9)))
  target_columns = tuple(config_value(args.target_columns, payload, "target_columns", tuple(build_target_columns(horizons))))
  auxiliary_target_columns = tuple(config_value(args.auxiliary_target_columns, payload, "auxiliary_target_columns", ()))

  config = HybridTransformerWalkForwardConfig(
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "data/models")),
    providers=tuple(config_value(args.providers, payload, "providers", ())),
    symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
    dataset_name=config_value(args.dataset_name, payload, "dataset_name", "market_supervised"),
    target_columns=target_columns,
    auxiliary_target_columns=auxiliary_target_columns,
    feature_groups=tuple(config_value(args.feature_groups, payload, "feature_groups", ())),
    exclude_feature_groups=tuple(config_value(args.exclude_feature_groups, payload, "exclude_feature_groups", ())),
    feature_prefixes=tuple(config_value(args.feature_prefixes, payload, "feature_prefixes", ())),
    exclude_feature_prefixes=tuple(config_value(args.exclude_feature_prefixes, payload, "exclude_feature_prefixes", ())),
    include_feature_columns=tuple(config_value(args.include_feature_columns, payload, "include_feature_columns", ())),
    exclude_feature_columns=tuple(config_value(args.exclude_feature_columns, payload, "exclude_feature_columns", ())),
    initial_train_rows=int(config_value(args.initial_train_rows, payload, "initial_train_rows", 512)),
    validation_rows=int(config_value(args.validation_rows, payload, "validation_rows", 128)),
    step_rows=_optional_int(args.step_rows if args.step_rows is not None else payload.get("step_rows")),
    embargo_rows=int(config_value(args.embargo_rows, payload, "embargo_rows", 0)),
    purge_rows=_optional_int(config_value(args.purge_rows, payload, "purge_rows")),
    minimum_rows=int(config_value(args.minimum_rows, payload, "minimum_rows", 768)),
    config_path=args.config,
    sample_weight_scheme=str(config_value(args.sample_weight_scheme, payload, "sample_weight_scheme", "")),
    sample_weight_target_column=str(config_value(args.sample_weight_target_column, payload, "sample_weight_target_column", "")),
    trend_column=str(config_value(args.trend_column, payload, "trend_column", "trend_return_24_bar")),
    trend_threshold_bps=float(config_value(args.trend_threshold_bps, payload, "trend_threshold_bps", 20.0)),
    regime_column=str(config_value(args.regime_column, payload, "regime_column", "regime_hmm_state_index")),
    regime_change_column=str(config_value(args.regime_change_column, payload, "regime_change_column", "regime_bocpd_change_prob")),
    regime_change_threshold=float(config_value(args.regime_change_threshold, payload, "regime_change_threshold", 0.35)),
    regime_change_boost=float(config_value(args.regime_change_boost, payload, "regime_change_boost", 1.0)),
    sample_weight_floor=float(config_value(args.sample_weight_floor, payload, "sample_weight_floor", 0.5)),
    sample_weight_cap=float(config_value(args.sample_weight_cap, payload, "sample_weight_cap", 3.0)),
    transformer=HybridTransformerConfig(
      lookback=int(config_value(args.lookback, transformer_payload, "lookback", 48)),
      backbone=str(config_value(args.backbone, transformer_payload, "backbone", "timemixer")),
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
      horizons=horizons,
      quantiles=quantiles,
      scales=tuple(config_value(args.scales, transformer_payload, "scales", (1, 2, 4))),
      mixer_layers=int(config_value(args.mixer_layers, transformer_payload, "mixer_layers", 2)),
      auxiliary_loss_weight=float(config_value(args.auxiliary_loss_weight, transformer_payload, "auxiliary_loss_weight", 0.25)),
      auxiliary_loss_weights=_auxiliary_loss_weights(args.auxiliary_loss_weights, transformer_payload),
    ),
  )

  summary = evaluate_hybrid_transformer_walk_forward(config)
  logging.getLogger(__name__).info(
    "Hybrid walk-forward complete. runs_attempted=%s runs_completed=%s folds_completed=%s artifacts_written=%s",
    summary.runs_attempted,
    summary.runs_completed,
    summary.folds_completed,
    summary.artifacts_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run purged walk-forward validation for the hybrid quantile transformer.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--dataset-name")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--target-column", dest="target_columns", action="append")
  parser.add_argument("--auxiliary-target-column", dest="auxiliary_target_columns", action="append")
  parser.add_argument("--feature-group", dest="feature_groups", action="append")
  parser.add_argument("--exclude-feature-group", dest="exclude_feature_groups", action="append")
  parser.add_argument("--feature-prefix", dest="feature_prefixes", action="append")
  parser.add_argument("--exclude-feature-prefix", dest="exclude_feature_prefixes", action="append")
  parser.add_argument("--feature-column", dest="include_feature_columns", action="append")
  parser.add_argument("--exclude-feature-column", dest="exclude_feature_columns", action="append")
  parser.add_argument("--horizon", dest="horizons", type=int, action="append")
  parser.add_argument("--quantile", dest="quantiles", type=float, action="append")
  parser.add_argument("--initial-train-rows", type=int)
  parser.add_argument("--validation-rows", type=int)
  parser.add_argument("--step-rows", type=int)
  parser.add_argument("--embargo-rows", type=int)
  parser.add_argument("--purge-rows", type=int)
  parser.add_argument("--minimum-rows", type=int)
  parser.add_argument("--sample-weight-scheme")
  parser.add_argument("--sample-weight-target-column")
  parser.add_argument("--trend-column")
  parser.add_argument("--trend-threshold-bps", type=float)
  parser.add_argument("--regime-column")
  parser.add_argument("--regime-change-column")
  parser.add_argument("--regime-change-threshold", type=float)
  parser.add_argument("--regime-change-boost", type=float)
  parser.add_argument("--sample-weight-floor", type=float)
  parser.add_argument("--sample-weight-cap", type=float)
  parser.add_argument("--lookback", type=int)
  parser.add_argument("--backbone")
  parser.add_argument("--d-model", type=int)
  parser.add_argument("--nhead", type=int)
  parser.add_argument("--num-layers", type=int)
  parser.add_argument("--dim-feedforward", type=int)
  parser.add_argument("--scale", dest="scales", type=int, action="append")
  parser.add_argument("--mixer-layers", type=int)
  parser.add_argument("--auxiliary-loss-weight", type=float)
  parser.add_argument("--auxiliary-loss-weight-by-target", dest="auxiliary_loss_weights", action="append")
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


def _auxiliary_loss_weights(overrides: list[str] | None, transformer_payload: dict[str, object]) -> dict[str, float]:
  values = dict(transformer_payload.get("auxiliary_loss_weights", {}) or {})
  for override in overrides or ():
    if "=" not in override:
      raise ValueError("--auxiliary-loss-weight-by-target must use target_column=weight")
    target_column, weight = override.split("=", 1)
    values[target_column] = float(weight)
  return {str(target_column): float(weight) for target_column, weight in values.items()}
