from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from quant.models.transformers.hybrid_forecaster import HybridTransformerConfig, build_target_columns
from quant.research.layered_feature_ablation import (
  DEFAULT_META_PARAMS,
  DEFAULT_PRUNED_STRUCTURE_COLUMNS,
  LayeredAblationCase,
  LayeredFeatureAblationConfig,
  run_layered_feature_ablation,
)
from quant.risk.sizing import PositionSizingConfig
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  transformer_payload = payload.get("transformer", {})
  horizons = tuple(transformer_payload.get("horizons", (1, 6, 24, 96)))
  cases = tuple(
    LayeredAblationCase(
      name=row["name"],
      supervised_feature_groups=tuple(row.get("supervised_feature_groups", ())),
      supervised_exclude_feature_groups=tuple(row.get("supervised_exclude_feature_groups", ())),
      supervised_feature_prefixes=tuple(row.get("supervised_feature_prefixes", ())),
      supervised_exclude_feature_prefixes=tuple(row.get("supervised_exclude_feature_prefixes", ())),
      supervised_include_feature_columns=tuple(row.get("supervised_include_feature_columns", ())),
      supervised_exclude_feature_columns=tuple(row.get("supervised_exclude_feature_columns", ())),
      meta_feature_groups=tuple(row.get("meta_feature_groups", ())),
      meta_exclude_feature_groups=tuple(row.get("meta_exclude_feature_groups", ())),
      meta_feature_prefixes=tuple(row.get("meta_feature_prefixes", ())),
      meta_exclude_feature_prefixes=tuple(row.get("meta_exclude_feature_prefixes", ())),
      meta_include_feature_columns=tuple(row.get("meta_include_feature_columns", ())),
      meta_exclude_feature_columns=tuple(row.get("meta_exclude_feature_columns", ())),
    )
    for row in payload.get("cases", _default_cases_payload())
  )

  config = LayeredFeatureAblationConfig(
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "experiments/feature_ablation")),
    provider=config_value(args.provider, payload, "provider", "coinbase"),
    symbol=config_value(args.symbol, payload, "symbol", "BTC-USD"),
    supervised_dataset_name=payload.get("supervised_dataset_name", "market_supervised"),
    output_dataset_prefix=payload.get("output_dataset_prefix", "market_hybrid_stacked_ablation"),
    target_columns=tuple(payload.get("target_columns", tuple(build_target_columns(horizons)))),
    transformer=HybridTransformerConfig(
      lookback=int(transformer_payload.get("lookback", 32)),
      d_model=int(transformer_payload.get("d_model", 32)),
      nhead=int(transformer_payload.get("nhead", 4)),
      num_layers=int(transformer_payload.get("num_layers", 1)),
      dim_feedforward=int(transformer_payload.get("dim_feedforward", 192)),
      dropout=float(transformer_payload.get("dropout", 0.1216)),
      batch_size=int(transformer_payload.get("batch_size", 32)),
      epochs=int(transformer_payload.get("epochs", 4)),
      learning_rate=float(transformer_payload.get("learning_rate", 0.0006168722008562335)),
      weight_decay=float(transformer_payload.get("weight_decay", 3.0399568928322742e-05)),
      validation_fraction=float(transformer_payload.get("validation_fraction", 0.25)),
      device=config_value(args.device, transformer_payload, "device", "cpu"),
      horizons=horizons,
      quantiles=tuple(transformer_payload.get("quantiles", (0.1, 0.5, 0.9))),
    ),
    initial_train_rows=int(payload.get("initial_train_rows", 1024)),
    validation_rows=int(payload.get("validation_rows", 256)),
    step_rows=int(payload.get("step_rows", 256)),
    embargo_rows=int(payload.get("embargo_rows", 1)),
    purge_rows=int(payload.get("purge_rows", 96)),
    minimum_rows=int(payload.get("minimum_rows", 2048)),
    meta_validation_fraction=float(payload.get("meta_validation_fraction", 0.25)),
    meta_params=dict(payload.get("meta_params", {})) or dict(DEFAULT_META_PARAMS),
    fee_bps=float(payload.get("fee_bps", 1.0)),
    slippage_bps=float(payload.get("slippage_bps", 0.5)),
    sizing=PositionSizingConfig(**payload.get("sizing", {})),
    cases=cases,
  )

  summary = run_layered_feature_ablation(config)
  logging.getLogger(__name__).info(
    "Layered feature ablation complete. run_id=%s best_case_by_net_return=%s cases=%s",
    summary.run_id,
    summary.best_case_by_net_return,
    len(summary.cases),
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run recent-window ablations for layered structure/trend features.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--provider")
  parser.add_argument("--symbol")
  parser.add_argument("--device")
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


def _default_cases_payload() -> list[dict[str, Any]]:
  return [
    {
      "name": "baseline_core",
      "supervised_feature_groups": ["baseline_core"],
      "meta_feature_groups": ["baseline_core", "hybrid_transformer_oof"],
    },
    {
      "name": "trend_layer",
      "supervised_feature_groups": ["baseline_core", "trend_layer"],
      "meta_feature_groups": ["baseline_core", "trend_layer", "hybrid_transformer_oof"],
    },
    {
      "name": "structure_pruned",
      "supervised_feature_groups": ["baseline_core"],
      "supervised_include_feature_columns": list(DEFAULT_PRUNED_STRUCTURE_COLUMNS),
      "meta_feature_groups": ["baseline_core", "hybrid_transformer_oof"],
      "meta_include_feature_columns": list(DEFAULT_PRUNED_STRUCTURE_COLUMNS),
    },
    {
      "name": "trend_structure_pruned",
      "supervised_feature_groups": ["baseline_core", "trend_layer"],
      "supervised_include_feature_columns": list(DEFAULT_PRUNED_STRUCTURE_COLUMNS),
      "meta_feature_groups": ["baseline_core", "trend_layer", "hybrid_transformer_oof"],
      "meta_include_feature_columns": list(DEFAULT_PRUNED_STRUCTURE_COLUMNS),
    },
  ]
