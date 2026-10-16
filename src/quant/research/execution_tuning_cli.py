from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.research.execution_tuning import ExecutionTuningConfig, run_execution_tuning
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  config = ExecutionTuningConfig(
    rolling_holdout_root=Path(config_value(args.rolling_holdout_root, payload, "rolling_holdout_root")),
    output_root=Path(config_value(args.output_root, payload, "output_root")),
    model_names=tuple(config_value(args.model_names, payload, "model_names", ())),
    top_k=int(args.top_k if args.top_k is not None else payload.get("top_k", 5)),
  )
  summary = run_execution_tuning(config)
  logging.getLogger(__name__).info(
    "Execution tuning complete. output_root=%s models=%s",
    summary["output_root"],
    sorted(summary["model_results"]),
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Tune execution-only sizing parameters on saved rolling holdout evaluation artifacts.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--rolling-holdout-root")
  parser.add_argument("--output-root")
  parser.add_argument("--model-name", dest="model_names", action="append")
  parser.add_argument("--top-k", type=int)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
