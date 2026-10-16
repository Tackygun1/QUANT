from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.research.profit_per_trade_tuning import (
  ProfitPerTradeRequirements,
  ProfitPerTradeSweepConfig,
  run_profit_per_trade_sweep,
)
from quant.utils.config import load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
  payload = load_json_config(args.config)
  requirements_payload = payload.get("requirements", {})
  config = ProfitPerTradeSweepConfig(
    reference_artifact_dir=Path(payload["reference_artifact_dir"]),
    reference_backtest_dir=Path(payload["reference_backtest_dir"]),
    output_root=Path(payload["output_root"]),
    margin_thresholds=tuple(float(value) for value in payload.get("margin_thresholds", (0.12, 0.16, 0.2, 0.24))),
    minimum_hold_bars=tuple(int(value) for value in payload.get("minimum_hold_bars", (2, 6, 12))),
    minimum_rebalance_deltas=tuple(float(value) for value in payload.get("minimum_rebalance_deltas", (0.025, 0.05, 0.1))),
    fixed_target_positions=tuple(float(value) for value in payload.get("fixed_target_positions", (0.0, 0.25))),
    forecast_gate_profiles=tuple(
      (bool(item["enabled"]), float(item["fast_threshold_bps"]), float(item["slow_threshold_bps"]))
      for item in payload.get("forecast_gate_profiles", (
        {"enabled": False, "fast_threshold_bps": 3.0, "slow_threshold_bps": 5.0},
        {"enabled": True, "fast_threshold_bps": 3.0, "slow_threshold_bps": 5.0},
        {"enabled": True, "fast_threshold_bps": 5.0, "slow_threshold_bps": 8.0},
        {"enabled": True, "fast_threshold_bps": 5.0, "slow_threshold_bps": 10.0},
      ))
    ),
    requirements=ProfitPerTradeRequirements(
      minimum_entries=int(requirements_payload.get("minimum_entries", 100)),
      minimum_traded_folds=int(requirements_payload.get("minimum_traded_folds", 12)),
      minimum_positive_folds=int(requirements_payload.get("minimum_positive_folds", 9)),
      minimum_median_fold_return=float(requirements_payload.get("minimum_median_fold_return", 0.0)),
      maximum_best_fold_profit_share=float(requirements_payload.get("maximum_best_fold_profit_share", 0.4)),
      maximum_drawdown=float(requirements_payload.get("maximum_drawdown", 0.01)),
      stress_total_cost_bps=float(requirements_payload.get("stress_total_cost_bps", 3.0)),
      minimum_stress_net_return=float(requirements_payload.get("minimum_stress_net_return", 0.0)),
      minimum_net_edge_per_turnover_bps=float(requirements_payload.get("minimum_net_edge_per_turnover_bps", 2.5)),
    ),
    stress_fallback_candidates=int(payload.get("stress_fallback_candidates", 20)),
    config_path=args.config,
  )
  summary = run_profit_per_trade_sweep(config)
  logging.getLogger(__name__).info(
    "Profit-per-trade sweep complete. status=%s feasible=%s output=%s",
    summary["selection_status"],
    summary["feasible_candidate_count"],
    summary["output_dir"],
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run a hard-constrained OOF execution sweep targeting higher profit per trade."
  )
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--log-level", default="INFO")
  return parser


if __name__ == "__main__":
  raise SystemExit(main())
