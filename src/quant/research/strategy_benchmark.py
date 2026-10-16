from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest
from quant.backtesting.strategy_backtest import StrategyBacktestConfig, run_strategy_backtest
from quant.risk.sizing import PositionSizingConfig
from quant.strategy.basic_strategies import available_basic_strategies


DEFAULT_BENCHMARK_SIZING = PositionSizingConfig(
  volatility_column="rolling_return_std_5_bar",
  volatility_target=0.0015,
  volatility_floor=0.0005,
  max_abs_position=1.0,
  signal_threshold_bps=0.5,
  signal_scale_bps=5.0,
  max_trade_risk_fraction=0.01,
  stop_loss_sigma=2.0,
  stop_loss_floor_bps=100.0,
  max_position_change=0.5,
  max_daily_loss_fraction=0.02,
  max_portfolio_drawdown_fraction=0.1,
)


@dataclass(frozen=True, slots=True)
class StrategyBenchmarkConfig:
  artifact_dir: Path
  input_root: Path
  output_root: Path = Path("reports/strategy_benchmarks")
  strategy_names: tuple[str, ...] = available_basic_strategies()
  strategy_params: dict[str, dict[str, Any]] | None = None
  dataset_name: str | None = None
  realized_return_column: str = "forward_return_1_bar"
  fee_bps: float = 1.0
  slippage_bps: float = 0.5
  sizing: PositionSizingConfig = DEFAULT_BENCHMARK_SIZING
  config_path: Path | None = None


@dataclass(slots=True)
class StrategyBenchmarkSummary:
  benchmark_dir: Path
  model_summary_path: Path
  comparison_summary_path: Path
  best_entry_name: str
  best_entry_type: str


def run_strategy_benchmark(config: StrategyBenchmarkConfig) -> StrategyBenchmarkSummary:
  benchmark_dir = config.output_root / config.artifact_dir.parent.name / config.artifact_dir.name
  benchmark_dir.mkdir(parents=True, exist_ok=True)

  model_summary = run_model_backtest(
    ModelBacktestConfig(
      artifact_dir=config.artifact_dir,
      input_root=config.input_root,
      output_root=benchmark_dir / "model_backtests",
      dataset_name=config.dataset_name,
      realized_return_column=config.realized_return_column,
      fee_bps=config.fee_bps,
      slippage_bps=config.slippage_bps,
      sizing=config.sizing,
      config_path=config.config_path,
    )
  )
  model_summary_path = model_summary.output_dir / "summary.json" if model_summary.output_dir else benchmark_dir / "model_summary.json"
  model_payload = json.loads(model_summary_path.read_text(encoding="utf-8"))

  leaderboard: list[dict[str, Any]] = [
    {
      "name": model_payload["model_name"],
      "entry_type": "model",
      "summary_path": str(model_summary_path),
      "total_net_return": model_payload["total_net_return"],
      "max_drawdown": model_payload["max_drawdown"],
      "sharpe_like": model_payload["sharpe_like"],
      "trades": model_payload["trades"],
      "joined_rows": model_payload["joined_rows"],
    }
  ]

  for strategy_name in config.strategy_names:
    strategy_summary = run_strategy_backtest(
      StrategyBacktestConfig(
        strategy_name=strategy_name,
        input_root=config.input_root,
        output_root=benchmark_dir / "strategy_backtests",
        dataset_name=config.dataset_name or model_payload["dataset_name"],
        reference_artifact_dir=config.artifact_dir,
        strategy_params=(config.strategy_params or {}).get(strategy_name),
        realized_return_column=config.realized_return_column,
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
        sizing=config.sizing,
        config_path=config.config_path,
      )
    )
    strategy_summary_path = strategy_summary.output_dir / "summary.json" if strategy_summary.output_dir else benchmark_dir / f"{strategy_name}.json"
    strategy_payload = json.loads(strategy_summary_path.read_text(encoding="utf-8"))
    leaderboard.append(
      {
        "name": strategy_name,
        "entry_type": "strategy",
        "summary_path": str(strategy_summary_path),
        "total_net_return": strategy_payload["total_net_return"],
        "max_drawdown": strategy_payload["max_drawdown"],
        "sharpe_like": strategy_payload["sharpe_like"],
        "trades": strategy_payload["trades"],
        "joined_rows": strategy_payload["joined_rows"],
      }
    )

  leaderboard.sort(
    key=lambda row: (
      float(row["total_net_return"]),
      -float(row["max_drawdown"]),
      float(row["sharpe_like"]),
    ),
    reverse=True,
  )

  comparison_summary_path = benchmark_dir / "comparison_summary.json"
  comparison_payload = {
    "artifact_dir": str(config.artifact_dir),
    "input_root": str(config.input_root),
    "dataset_name": config.dataset_name or model_payload["dataset_name"],
    "realized_return_column": config.realized_return_column,
    "fee_bps": config.fee_bps,
    "slippage_bps": config.slippage_bps,
    "sizing": asdict(config.sizing),
    "leaderboard": leaderboard,
    "model_name": model_payload["model_name"],
    "best_entry": leaderboard[0] if leaderboard else {},
    "config_path": str(config.config_path) if config.config_path else "",
  }
  comparison_summary_path.write_text(
    json.dumps(comparison_payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )

  return StrategyBenchmarkSummary(
    benchmark_dir=benchmark_dir,
    model_summary_path=model_summary_path,
    comparison_summary_path=comparison_summary_path,
    best_entry_name=str(leaderboard[0]["name"]),
    best_entry_type=str(leaderboard[0]["entry_type"]),
  )
