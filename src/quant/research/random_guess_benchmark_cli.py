from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.research.random_guess_benchmark import RandomGuessBenchmarkConfig, run_random_guess_benchmark
from quant.utils.config import load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
  payload = load_json_config(args.config)
  summary = run_random_guess_benchmark(RandomGuessBenchmarkConfig(
    reference_artifact_dir=Path(payload["reference_artifact_dir"]),
    reference_backtest_dir=Path(payload["reference_backtest_dir"]),
    output_root=Path(payload["output_root"]),
    simulations_per_baseline=int(payload.get("simulations_per_baseline", 500)),
    seed=int(payload.get("seed", 20260818)),
    config_path=args.config,
  ))
  logging.getLogger(__name__).info(
    "Random baseline benchmark complete. status=%s simulations=%s output=%s",
    summary.status,
    summary.simulations_completed,
    summary.output_dir,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Compare a classification artifact with random-guess and shuffled-signal baselines."
  )
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--log-level", default="INFO")
  return parser


if __name__ == "__main__":
  raise SystemExit(main())
