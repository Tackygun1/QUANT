from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import (
  PERIOD_FIELDS,
  TRADE_FIELDS,
  simulate_prediction_backtest,
)
from quant.models.base import utc_run_slug
from quant.risk.sizing import PositionSizingConfig
from quant.strategy.basic_strategies import basic_strategy_prediction, resolve_strategy_params
from quant.training.datasets import SeriesKey, load_series_rows_for_dataset


@dataclass(frozen=True, slots=True)
class StrategyBacktestConfig:
  strategy_name: str
  provider: str = ""
  symbol: str = ""
  input_root: Path = Path("data/features")
  output_root: Path = Path("reports/strategy_backtests")
  dataset_name: str = "market_supervised"
  reference_artifact_dir: Path | None = None
  strategy_params: dict[str, Any] | None = None
  realized_return_column: str = "forward_return_1_bar"
  fee_bps: float = 5.0
  slippage_bps: float = 1.0
  sizing: PositionSizingConfig = PositionSizingConfig()
  config_path: Path | None = None


@dataclass(slots=True)
class StrategyBacktestSummary:
  status: str = "skipped"
  output_dir: Path | None = None
  joined_rows: int = 0
  active_periods: int = 0
  trades: int = 0
  total_net_return: float = 0.0
  max_drawdown: float = 0.0
  sharpe_like: float = 0.0


def run_strategy_backtest(config: StrategyBacktestConfig) -> StrategyBacktestSummary:
  provider, symbol, dataset_name, reference_metadata, reference_candle_order = _resolve_reference_context(config)
  strategy_params = resolve_strategy_params(config.strategy_name, config.strategy_params)
  series_key = SeriesKey(provider=provider, symbol=symbol)
  dataset_rows = load_series_rows_for_dataset(
    config.input_root,
    series_key,
    dataset_name=dataset_name,
  )

  selected_rows = _filter_rows_to_reference(dataset_rows, reference_candle_order)
  joined_rows: list[dict[str, str]] = []
  for row in selected_rows:
    realized_return = row.get(config.realized_return_column, "")
    if realized_return == "":
      continue
    joined = dict(row)
    joined["prediction"] = str(basic_strategy_prediction(config.strategy_name, row, strategy_params))
    joined_rows.append(joined)

  output_dir = (
    config.output_root
    / config.strategy_name
    / provider
    / symbol
    / utc_run_slug()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  if not joined_rows:
    summary = StrategyBacktestSummary(status="skipped", output_dir=output_dir)
    _write_summary(
      output_dir / "summary.json",
      summary,
      config=config,
      provider=provider,
      symbol=symbol,
      dataset_name=dataset_name,
      strategy_params=strategy_params,
      aggregate_metrics={},
      reference_metadata=reference_metadata,
    )
    _write_csv(output_dir / "periods.csv", [], PERIOD_FIELDS)
    _write_csv(output_dir / "trades.csv", [], TRADE_FIELDS)
    return summary

  period_rows, trade_rows, aggregate_metrics = simulate_prediction_backtest(
    rows=joined_rows,
    task_type="regression",
    realized_return_column=config.realized_return_column,
    fee_bps=config.fee_bps,
    slippage_bps=config.slippage_bps,
    sizing=config.sizing,
  )
  summary = StrategyBacktestSummary(
    status="completed",
    output_dir=output_dir,
    joined_rows=len(joined_rows),
    active_periods=int(aggregate_metrics.get("active_periods", 0)),
    trades=int(aggregate_metrics.get("trade_count", 0)),
    total_net_return=float(aggregate_metrics.get("total_net_return", 0.0)),
    max_drawdown=float(aggregate_metrics.get("max_drawdown", 0.0)),
    sharpe_like=float(aggregate_metrics.get("sharpe_like", 0.0)),
  )
  _write_summary(
    output_dir / "summary.json",
    summary,
    config=config,
    provider=provider,
    symbol=symbol,
    dataset_name=dataset_name,
    strategy_params=strategy_params,
    aggregate_metrics=aggregate_metrics,
    reference_metadata=reference_metadata,
  )
  _write_csv(output_dir / "periods.csv", period_rows, PERIOD_FIELDS)
  _write_csv(output_dir / "trades.csv", trade_rows, TRADE_FIELDS)
  (output_dir / "backtest_config.json").write_text(
    json.dumps(
      {
        "strategy_name": config.strategy_name,
        "provider": provider,
        "symbol": symbol,
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "dataset_name": dataset_name,
        "strategy_params": strategy_params,
        "reference_artifact_dir": str(config.reference_artifact_dir) if config.reference_artifact_dir else "",
        "realized_return_column": config.realized_return_column,
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "sizing": asdict(config.sizing),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return summary


def _resolve_reference_context(
  config: StrategyBacktestConfig,
) -> tuple[str, str, str, dict[str, Any], list[str]]:
  reference_metadata: dict[str, Any] = {}
  reference_candle_order: list[str] = []
  provider = config.provider
  symbol = config.symbol
  dataset_name = config.dataset_name

  if config.reference_artifact_dir is not None:
    metadata_path = config.reference_artifact_dir / "metadata.json"
    predictions_path = config.reference_artifact_dir / "validation_predictions.csv"
    if metadata_path.exists():
      reference_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      provider = provider or reference_metadata.get("provider", "")
      symbol = symbol or reference_metadata.get("symbol", "")
      dataset_name = reference_metadata.get("extra", {}).get("dataset_name", dataset_name)
    if predictions_path.exists():
      with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reference_candle_order = [
          row.get("candle_start", "")
          for row in csv.DictReader(handle)
          if row.get("candle_start", "")
        ]

  if provider == "" or symbol == "":
    raise ValueError("Strategy backtest requires provider and symbol, either directly or via reference_artifact_dir.")
  return provider, symbol, dataset_name, reference_metadata, reference_candle_order


def _filter_rows_to_reference(rows: list[dict[str, str]], reference_candle_order: list[str]) -> list[dict[str, str]]:
  if not reference_candle_order:
    return rows
  lookup = {row.get("candle_start", ""): row for row in rows}
  return [lookup[candle_start] for candle_start in reference_candle_order if candle_start in lookup]


def _write_summary(
  path: Path,
  summary: StrategyBacktestSummary,
  *,
  config: StrategyBacktestConfig,
  provider: str,
  symbol: str,
  dataset_name: str,
  strategy_params: dict[str, Any],
  aggregate_metrics: dict[str, float],
  reference_metadata: dict[str, Any],
) -> None:
  payload = {
    "status": summary.status,
    "strategy_name": config.strategy_name,
    "provider": provider,
    "symbol": symbol,
    "dataset_name": dataset_name,
    "reference_artifact_dir": str(config.reference_artifact_dir) if config.reference_artifact_dir else "",
    "reference_model_name": reference_metadata.get("model_name", ""),
    "realized_return_column": config.realized_return_column,
    "joined_rows": summary.joined_rows,
    "active_periods": summary.active_periods,
    "trades": summary.trades,
    "total_net_return": summary.total_net_return,
    "max_drawdown": summary.max_drawdown,
    "sharpe_like": summary.sharpe_like,
    "strategy_params": strategy_params,
    "fee_bps": config.fee_bps,
    "slippage_bps": config.slippage_bps,
    "sizing": asdict(config.sizing),
    "aggregate_metrics": aggregate_metrics,
    "config_path": str(config.config_path) if config.config_path else "",
  }
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
