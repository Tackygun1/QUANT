from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant.models.base import utc_run_slug
from quant.risk.sizing import (
  PositionSizingConfig,
  RiskState,
  mark_position_state,
  next_active_position_bars,
  next_position_state,
  target_position_from_prediction,
)
from quant.training.datasets import SeriesKey, load_series_rows_for_dataset


@dataclass(frozen=True, slots=True)
class ModelBacktestConfig:
  artifact_dir: Path
  input_root: Path = Path("data/features")
  output_root: Path = Path("reports/backtests")
  dataset_name: str | None = None
  realized_return_column: str = "forward_return_1_bar"
  fee_bps: float = 5.0
  slippage_bps: float = 1.0
  sizing: PositionSizingConfig = PositionSizingConfig()
  config_path: Path | None = None


@dataclass(slots=True)
class ModelBacktestSummary:
  status: str = "skipped"
  output_dir: Path | None = None
  joined_rows: int = 0
  active_periods: int = 0
  trades: int = 0
  total_net_return: float = 0.0
  max_drawdown: float = 0.0
  sharpe_like: float = 0.0
  total_transaction_cost: float = 0.0
  total_turnover: float = 0.0
  active_hit_rate: float = 0.0
  average_net_return_per_trade: float = 0.0
  profit_factor_after_cost: float = 0.0
  after_cost_trade_quality: float = 0.0


def run_model_backtest(config: ModelBacktestConfig) -> ModelBacktestSummary:
  metadata = _load_json(config.artifact_dir / "metadata.json")
  prediction_rows = _load_csv(config.artifact_dir / "validation_predictions.csv")
  dataset_name = config.dataset_name or metadata.get("extra", {}).get("dataset_name", "market_supervised")
  prediction_column = metadata.get("extra", {}).get("primary_prediction_column", "prediction")
  series_key = SeriesKey(
    provider=metadata["provider"],
    symbol=metadata["symbol"],
  )
  dataset_rows = load_series_rows_for_dataset(
    config.input_root,
    series_key,
    dataset_name=dataset_name,
  )
  row_lookup = {row.get("candle_start", ""): row for row in dataset_rows}

  joined_rows = []
  for prediction_row in prediction_rows:
    candle_start = prediction_row.get("candle_start", "")
    dataset_row = row_lookup.get(candle_start)
    if dataset_row is None:
      continue
    realized_return = dataset_row.get(config.realized_return_column, "")
    prediction = prediction_row.get(prediction_column, prediction_row.get("prediction", ""))
    if realized_return == "" or prediction == "":
      continue
    joined = dict(dataset_row)
    joined["prediction"] = prediction
    joined["prediction_target"] = prediction_row.get("target", "")
    joined_rows.append(joined)

  joined_rows.sort(key=lambda row: row.get("candle_start", ""))
  output_dir = (
    config.output_root
    / metadata["model_name"]
    / metadata["provider"]
    / metadata["symbol"]
    / utc_run_slug()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  if not joined_rows:
    summary = ModelBacktestSummary(status="skipped", output_dir=output_dir)
    _write_summary(
      output_dir / "summary.json",
      summary,
      config=config,
      metadata=metadata,
      dataset_name=dataset_name,
      aggregate_metrics={},
    )
    _write_csv(output_dir / "periods.csv", [], PERIOD_FIELDS)
    _write_csv(output_dir / "trades.csv", [], TRADE_FIELDS)
    return summary

  period_rows, trade_rows, aggregate_metrics = simulate_prediction_backtest(
    rows=joined_rows,
    task_type=metadata["task_type"],
    realized_return_column=config.realized_return_column,
    fee_bps=config.fee_bps,
    slippage_bps=config.slippage_bps,
    sizing=config.sizing,
  )

  summary = ModelBacktestSummary(
    status="completed",
    output_dir=output_dir,
    joined_rows=len(joined_rows),
    active_periods=int(aggregate_metrics.get("active_periods", 0)),
    trades=int(aggregate_metrics.get("trade_count", 0)),
    total_net_return=float(aggregate_metrics.get("total_net_return", 0.0)),
    max_drawdown=float(aggregate_metrics.get("max_drawdown", 0.0)),
    sharpe_like=float(aggregate_metrics.get("sharpe_like", 0.0)),
    total_transaction_cost=float(aggregate_metrics.get("total_transaction_cost", 0.0)),
    total_turnover=float(aggregate_metrics.get("total_turnover", 0.0)),
    active_hit_rate=float(aggregate_metrics.get("active_hit_rate", 0.0)),
    average_net_return_per_trade=float(aggregate_metrics.get("average_net_return_per_trade", 0.0)),
    profit_factor_after_cost=float(aggregate_metrics.get("profit_factor_after_cost", 0.0)),
    after_cost_trade_quality=float(aggregate_metrics.get("after_cost_trade_quality", 0.0)),
  )
  _write_summary(
    output_dir / "summary.json",
    summary,
    config=config,
    metadata=metadata,
    dataset_name=dataset_name,
    aggregate_metrics=aggregate_metrics,
  )
  _write_csv(output_dir / "periods.csv", period_rows, PERIOD_FIELDS)
  _write_csv(output_dir / "trades.csv", trade_rows, TRADE_FIELDS)
  (output_dir / "backtest_config.json").write_text(
    json.dumps(
      {
        "artifact_dir": str(config.artifact_dir),
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "dataset_name": dataset_name,
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


PERIOD_FIELDS = [
  "candle_start",
  "prediction",
  "realized_return",
  "signal",
  "direction",
  "confidence",
  "volatility",
  "volatility_scale",
  "raw_target_position",
  "target_position",
  "active_position_bars",
  "entry_price",
  "open_return",
  "max_favorable_return",
  "max_adverse_return",
  "pending_direction",
  "pending_direction_bars",
  "stop_distance_rate",
  "max_position_by_risk",
  "session_drawdown",
  "portfolio_drawdown",
  "constraint_flags",
  "turnover",
  "transaction_cost",
  "gross_return",
  "net_return",
  "equity",
]

TRADE_FIELDS = [
  "candle_start",
  "direction",
  "raw_target_position",
  "target_position",
  "active_position_bars",
  "entry_price",
  "open_return",
  "max_favorable_return",
  "pending_direction",
  "pending_direction_bars",
  "constraint_flags",
  "turnover",
  "transaction_cost",
  "prediction",
  "realized_return",
  "net_return",
]


def simulate_prediction_backtest(
  *,
  rows: list[dict[str, str]],
  task_type: str,
  realized_return_column: str,
  fee_bps: float,
  slippage_bps: float,
  sizing: PositionSizingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
  execution_cost_rate = (fee_bps + slippage_bps) / 10_000
  period_rows: list[dict[str, Any]] = []
  trade_rows: list[dict[str, Any]] = []
  net_returns: list[float] = []
  active_returns: list[float] = []
  equity = 1.0
  peak_equity = 1.0
  max_drawdown = 0.0
  previous_position = 0.0
  previous_active_position_bars = 0
  pending_direction = 0
  pending_direction_bars = 0
  current_session_date = ""
  session_start_equity = equity
  active_periods = 0
  trade_count = 0
  entry_count = 0
  flip_count = 0
  total_turnover = 0.0
  total_cost = 0.0
  total_gross_return = 0.0
  position_entry_price: float | None = None
  position_max_favorable_return = 0.0
  position_max_adverse_return = 0.0
  risk_flag_counts: dict[str, float] = {
    "risk_budget_cap_periods": 0.0,
    "turnover_cap_periods": 0.0,
    "daily_loss_halt_periods": 0.0,
    "drawdown_halt_periods": 0.0,
    "flip_exit_hysteresis_periods": 0.0,
    "forecast_gate_entry_periods": 0.0,
    "forecast_gate_reverse_periods": 0.0,
    "forecast_gate_scale_in_periods": 0.0,
    "hold_until_exit_periods": 0.0,
    "max_hold_exit_periods": 0.0,
    "regime_chop_entry_periods": 0.0,
    "regime_chop_reverse_periods": 0.0,
    "regime_chop_size_cap_periods": 0.0,
    "stale_exit_periods": 0.0,
    "stop_loss_exit_periods": 0.0,
    "target_smoothing_periods": 0.0,
    "time_stop_hold_periods": 0.0,
    "profit_take_exit_periods": 0.0,
    "trailing_profit_exit_periods": 0.0,
    "tradeability_gate_entry_periods": 0.0,
    "tradeability_gate_reverse_periods": 0.0,
    "tradeability_gate_scale_in_periods": 0.0,
    "trend_filter_exit_periods": 0.0,
    "trend_state_exit_periods": 0.0,
    "trend_state_chop_exit_periods": 0.0,
  }
  interval = rows[0].get("interval", "1m")

  for row in rows:
    candle_start = row.get("candle_start", "")
    session_date = candle_start[:10]
    if session_date and session_date != current_session_date:
      current_session_date = session_date
      session_start_equity = equity

    prediction = float(row["prediction"])
    realized_return = float(row[realized_return_column])
    position_state = mark_position_state(
      row=row,
      previous_position=previous_position,
      entry_price=position_entry_price,
      max_favorable_return=position_max_favorable_return,
      max_adverse_return=position_max_adverse_return,
    )
    decision = target_position_from_prediction(
      task_type=task_type,
      prediction=prediction,
      row=row,
      sizing=sizing,
      execution_cost_rate=execution_cost_rate,
      previous_position=previous_position,
      active_position_bars=previous_active_position_bars,
      pending_direction=pending_direction,
      pending_direction_bars=pending_direction_bars,
      risk_state=RiskState(
        equity=equity,
        peak_equity=peak_equity,
        session_start_equity=session_start_equity,
      ),
      position_state=position_state,
    )
    for flag in decision.constraint_flags:
      metric_key = f"{flag}_periods"
      if metric_key in risk_flag_counts:
        risk_flag_counts[metric_key] += 1.0

    turnover = abs(decision.target_position - previous_position)
    active_position_bars = next_active_position_bars(
      previous_position=previous_position,
      previous_active_position_bars=previous_active_position_bars,
      target_position=decision.target_position,
    )
    next_state = next_position_state(
      row=row,
      previous_position=previous_position,
      target_position=decision.target_position,
      previous_state=position_state,
    )
    transaction_cost = turnover * execution_cost_rate
    gross_return = decision.target_position * realized_return
    net_return = gross_return - transaction_cost
    equity *= (1.0 + net_return)
    peak_equity = max(peak_equity, equity)
    max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0)

    total_turnover += turnover
    total_cost += transaction_cost
    total_gross_return += gross_return
    net_returns.append(net_return)
    if abs(decision.target_position) > 0:
      active_periods += 1
      active_returns.append(net_return)
    if turnover > 0:
      trade_count += 1
      if previous_position == 0 and decision.target_position != 0:
        entry_count += 1
      if previous_position != 0 and decision.target_position != 0 and _sign(previous_position) != _sign(decision.target_position):
        flip_count += 1
      trade_rows.append(
        {
          "candle_start": candle_start,
          "direction": decision.direction,
          "raw_target_position": decision.raw_target_position,
          "target_position": decision.target_position,
          "active_position_bars": active_position_bars,
          "entry_price": next_state.entry_price,
          "open_return": next_state.open_return,
          "max_favorable_return": next_state.max_favorable_return,
          "pending_direction": decision.pending_direction,
          "pending_direction_bars": decision.pending_direction_bars,
          "constraint_flags": "|".join(decision.constraint_flags),
          "turnover": turnover,
          "transaction_cost": transaction_cost,
          "prediction": prediction,
          "realized_return": realized_return,
          "net_return": net_return,
        }
      )

    period_rows.append(
      {
        "candle_start": candle_start,
        "prediction": prediction,
        "realized_return": realized_return,
        "signal": decision.signal,
        "direction": decision.direction,
        "confidence": decision.confidence,
        "volatility": decision.volatility,
        "volatility_scale": decision.volatility_scale,
        "raw_target_position": decision.raw_target_position,
        "target_position": decision.target_position,
        "active_position_bars": active_position_bars,
        "entry_price": next_state.entry_price,
        "open_return": next_state.open_return,
        "max_favorable_return": next_state.max_favorable_return,
        "max_adverse_return": next_state.max_adverse_return,
        "pending_direction": decision.pending_direction,
        "pending_direction_bars": decision.pending_direction_bars,
        "stop_distance_rate": decision.stop_distance_rate,
        "max_position_by_risk": decision.max_position_by_risk,
        "session_drawdown": decision.session_drawdown,
        "portfolio_drawdown": decision.portfolio_drawdown,
        "constraint_flags": "|".join(decision.constraint_flags),
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "gross_return": gross_return,
        "net_return": net_return,
        "equity": equity,
      }
    )
    previous_position = decision.target_position
    previous_active_position_bars = active_position_bars
    pending_direction = decision.pending_direction
    pending_direction_bars = decision.pending_direction_bars
    position_entry_price = next_state.entry_price
    position_max_favorable_return = next_state.max_favorable_return
    position_max_adverse_return = next_state.max_adverse_return

  periods_per_year = _periods_per_year(interval)
  sharpe_like = _sharpe_like(net_returns, periods_per_year)
  active_hit_rate = (
    sum(1 for row in period_rows if float(row["target_position"]) != 0 and float(row["gross_return"]) > 0) / active_periods
    if active_periods
    else 0.0
  )
  trade_net_returns = [float(row["net_return"]) for row in trade_rows]
  positive_trade_return = sum(value for value in trade_net_returns if value > 0)
  negative_trade_return = abs(sum(value for value in trade_net_returns if value < 0))
  profit_factor_after_cost = (
    positive_trade_return / negative_trade_return
    if negative_trade_return > 0
    else (positive_trade_return if positive_trade_return > 0 else 0.0)
  )
  average_net_return_per_trade = (
    sum(trade_net_returns) / len(trade_net_returns)
    if trade_net_returns
    else 0.0
  )
  after_cost_trade_quality = (equity - 1.0) - max_drawdown - total_cost

  aggregate_metrics = {
    "periods": float(len(period_rows)),
    "active_periods": float(active_periods),
    "trade_count": float(trade_count),
    "entry_count": float(entry_count),
    "flip_count": float(flip_count),
    "total_turnover": total_turnover,
    "average_turnover_per_period": total_turnover / len(period_rows),
    "total_gross_return": total_gross_return,
    "total_net_return": equity - 1.0,
    "average_net_return": sum(net_returns) / len(net_returns),
    "average_active_return": sum(active_returns) / len(active_returns) if active_returns else 0.0,
    "total_transaction_cost": total_cost,
    "active_hit_rate": active_hit_rate,
    "average_net_return_per_trade": average_net_return_per_trade,
    "profit_factor_after_cost": profit_factor_after_cost,
    "after_cost_trade_quality": after_cost_trade_quality,
    "max_drawdown": max_drawdown,
    "sharpe_like": sharpe_like,
    "final_equity": equity,
  }
  aggregate_metrics.update(risk_flag_counts)
  return period_rows, trade_rows, aggregate_metrics


def _write_summary(
  path: Path,
  summary: ModelBacktestSummary,
  *,
  config: ModelBacktestConfig,
  metadata: dict[str, Any],
  dataset_name: str,
  aggregate_metrics: dict[str, float],
) -> None:
  payload = {
    "status": summary.status,
    "artifact_dir": str(config.artifact_dir),
    "dataset_name": dataset_name,
    "provider": metadata["provider"],
    "symbol": metadata["symbol"],
    "model_name": metadata["model_name"],
    "task_type": metadata["task_type"],
    "target_column": metadata["target_column"],
    "realized_return_column": config.realized_return_column,
    "joined_rows": summary.joined_rows,
    "active_periods": summary.active_periods,
    "trades": summary.trades,
    "total_net_return": summary.total_net_return,
    "max_drawdown": summary.max_drawdown,
    "sharpe_like": summary.sharpe_like,
    "total_transaction_cost": summary.total_transaction_cost,
    "total_turnover": summary.total_turnover,
    "active_hit_rate": summary.active_hit_rate,
    "average_net_return_per_trade": summary.average_net_return_per_trade,
    "profit_factor_after_cost": summary.profit_factor_after_cost,
    "after_cost_trade_quality": summary.after_cost_trade_quality,
    "fee_bps": config.fee_bps,
    "slippage_bps": config.slippage_bps,
    "sizing": asdict(config.sizing),
    "aggregate_metrics": aggregate_metrics,
    "config_path": str(config.config_path) if config.config_path else "",
  }
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})


def _periods_per_year(interval: str) -> float:
  if interval.endswith("m"):
    minutes = int(interval[:-1])
    return (60 / minutes) * 24 * 365
  if interval.endswith("h"):
    hours = int(interval[:-1])
    return (24 / hours) * 365
  if interval.endswith("d"):
    days = int(interval[:-1])
    return 365 / days
  return 1.0


def _sharpe_like(returns: list[float], periods_per_year: float) -> float:
  if len(returns) < 30:
    return 0.0
  mean_return = sum(returns) / len(returns)
  variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
  if variance <= 0:
    return 0.0
  return mean_return / math.sqrt(variance) * math.sqrt(periods_per_year)


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0
