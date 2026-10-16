from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant.research.frozen_holdout import FrozenHoldoutConfig, align_to_bar, align_to_next_bar, run_frozen_holdout_evaluation
from quant.research.model_weakness_diagnostics import (
  ModelWeaknessDiagnosticsConfig,
  run_model_weakness_diagnostics,
)
from quant.risk.sizing import PositionSizingConfig


@dataclass(frozen=True, slots=True)
class RollingWindow:
  index: int
  evaluation_start_time: datetime
  evaluation_end_time: datetime

  @property
  def slug(self) -> str:
    start = self.evaluation_start_time.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    end = self.evaluation_end_time.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{self.index:02d}_{start}_{end}"


@dataclass(frozen=True, slots=True)
class RollingHoldoutConfig:
  source_workspace_root: Path
  artifact_dirs: tuple[Path, ...]
  output_root: Path
  provider: str = "coinbase"
  symbol: str = "BTC-USD"
  candle_interval: str = "5m"
  evaluation_start_time: datetime | None = None
  evaluation_end_time: datetime | None = None
  window_hours: float = 2.0
  step_hours: float | None = None
  warmup_bars: int = 64
  request_timeout_seconds: float = 20.0
  request_pause_seconds: float = 0.02
  retry_attempts: int = 3
  retry_pause_seconds: float = 20.0
  fee_bps: float = 1.0
  slippage_bps: float = 0.5
  use_artifact_execution_sizing: bool = True
  sizing: PositionSizingConfig = PositionSizingConfig(
    signal_threshold_bps=0.5,
    signal_scale_bps=5.0,
  )
  trend_column: str = "trend_return_24_bar"
  trend_threshold_bps: float = 20.0
  clean_start: bool = True
  config_path: Path | None = None


@dataclass(slots=True)
class RollingHoldoutSummary:
  output_root: Path
  overall_start_time: datetime
  overall_end_time: datetime
  windows: list[dict[str, Any]] = field(default_factory=list)
  model_comparison: dict[str, dict[str, Any]] = field(default_factory=dict)


def run_rolling_holdout_evaluation(config: RollingHoldoutConfig) -> RollingHoldoutSummary:
  output_root = config.output_root
  if config.clean_start and output_root.exists():
    shutil.rmtree(output_root)
  output_root.mkdir(parents=True, exist_ok=True)

  overall_start, overall_end = resolve_rolling_range(config)
  windows = build_rolling_windows(
    evaluation_start=overall_start,
    evaluation_end=overall_end,
    window_hours=config.window_hours,
    step_hours=config.step_hours,
  )
  if not windows:
    raise ValueError(
      f"No full rolling windows fit inside {overall_start.isoformat()} to {overall_end.isoformat()}."
    )

  window_payloads: list[dict[str, Any]] = []
  per_model_windows: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for window in windows:
    holdout_root = output_root / "windows" / window.slug
    holdout_summary = run_frozen_holdout_with_retries(
      config=config,
      holdout_root=holdout_root,
      window=window,
    )

    model_results: dict[str, Any] = {}
    for model_name, result in holdout_summary.results.items():
      backtest_dir = Path(result["backtest"]["output_dir"])
      backtest_summary = json.loads((backtest_dir / "summary.json").read_text(encoding="utf-8"))
      weakness_summary = run_model_weakness_diagnostics(
        ModelWeaknessDiagnosticsConfig(
          backtest_dir=backtest_dir,
          feature_root=holdout_root / "features",
          provider=config.provider,
          symbol=config.symbol,
          dataset_name=str(backtest_summary.get("dataset_name", "market_supervised")),
          trend_column=config.trend_column,
          trend_threshold_bps=config.trend_threshold_bps,
        )
      )
      payload = {
        "model_name": model_name,
        "artifact_dir": result["source_artifact_dir"],
        "evaluation_artifact_dir": result["evaluation_artifact_dir"],
        "evaluation_rows": result["evaluation_rows"],
        "metrics": result["metrics"],
        "backtest": result["backtest"],
        "aggregate_metrics": backtest_summary.get("aggregate_metrics", {}),
        "weakness": _weakness_payload(weakness_summary),
      }
      model_results[model_name] = payload
      per_model_windows[model_name].append(payload)

    window_payloads.append(
      {
        "window_index": window.index,
        "evaluation_start_time": _format_dt(window.evaluation_start_time),
        "evaluation_end_time": _format_dt(window.evaluation_end_time),
        "evaluation_rows": holdout_summary.evaluation_rows,
        "holdout_root": str(holdout_root),
        "results": model_results,
      }
    )

  comparison = {
    model_name: aggregate_model_window_results(model_name, payloads)
    for model_name, payloads in per_model_windows.items()
  }
  summary = RollingHoldoutSummary(
    output_root=output_root,
    overall_start_time=overall_start,
    overall_end_time=overall_end,
    windows=window_payloads,
    model_comparison=comparison,
  )
  (output_root / "summary.json").write_text(
    json.dumps(
      {
        "output_root": str(summary.output_root),
        "overall_start_time": _format_dt(summary.overall_start_time),
        "overall_end_time": _format_dt(summary.overall_end_time),
        "window_count": len(summary.windows),
        "windows": summary.windows,
        "model_comparison": summary.model_comparison,
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return summary


def run_frozen_holdout_with_retries(
  *,
  config: RollingHoldoutConfig,
  holdout_root: Path,
  window: RollingWindow,
):
  attempts = 0
  last_error: Exception | None = None
  while attempts < config.retry_attempts:
    attempts += 1
    try:
      return run_frozen_holdout_evaluation(
        FrozenHoldoutConfig(
          source_workspace_root=config.source_workspace_root,
          artifact_dirs=config.artifact_dirs,
          holdout_root=holdout_root,
          provider=config.provider,
          symbol=config.symbol,
          candle_interval=config.candle_interval,
          evaluation_start_time=window.evaluation_start_time,
          evaluation_end_time=window.evaluation_end_time,
          warmup_bars=config.warmup_bars,
          request_timeout_seconds=config.request_timeout_seconds,
          request_pause_seconds=config.request_pause_seconds,
          fee_bps=config.fee_bps,
          slippage_bps=config.slippage_bps,
          use_artifact_execution_sizing=config.use_artifact_execution_sizing,
          sizing=config.sizing,
          clean_start=True,
          config_path=config.config_path,
        )
      )
    except Exception as exc:
      last_error = exc
      if holdout_root.exists():
        shutil.rmtree(holdout_root)
      if attempts >= config.retry_attempts:
        raise
      import time

      time.sleep(config.retry_pause_seconds)
  if last_error is not None:
    raise last_error
  raise RuntimeError("Rolling holdout retry loop exited unexpectedly.")


def resolve_rolling_range(config: RollingHoldoutConfig) -> tuple[datetime, datetime]:
  state_path = config.source_workspace_root / "state" / "overnight_state.json"
  state = json.loads(state_path.read_text(encoding="utf-8"))
  source_end = _parse_dt(state["dataset_window"]["end_time"])
  evaluation_start = config.evaluation_start_time or align_to_next_bar(source_end, config.candle_interval)
  evaluation_end = config.evaluation_end_time or align_to_bar(datetime.now(UTC), config.candle_interval)
  if evaluation_end <= evaluation_start:
    raise ValueError(
      f"Evaluation end {evaluation_end} must be after evaluation start {evaluation_start}."
    )
  return evaluation_start, evaluation_end


def build_rolling_windows(
  *,
  evaluation_start: datetime,
  evaluation_end: datetime,
  window_hours: float,
  step_hours: float | None,
) -> list[RollingWindow]:
  if window_hours <= 0:
    raise ValueError("window_hours must be positive.")
  step = step_hours if step_hours is not None else window_hours
  if step <= 0:
    raise ValueError("step_hours must be positive.")

  window_delta = timedelta(hours=window_hours)
  step_delta = timedelta(hours=step)
  windows: list[RollingWindow] = []
  cursor = evaluation_start.astimezone(UTC)
  end = evaluation_end.astimezone(UTC)
  index = 1
  while cursor + window_delta <= end:
    windows.append(
      RollingWindow(
        index=index,
        evaluation_start_time=cursor,
        evaluation_end_time=cursor + window_delta,
      )
    )
    cursor += step_delta
    index += 1
  return windows


def aggregate_model_window_results(model_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
  net_returns = [float(item["backtest"]["total_net_return"]) for item in results]
  max_drawdowns = [float(item["backtest"]["max_drawdown"]) for item in results]
  trades = [int(item["backtest"]["trades"]) for item in results]
  active_periods = [int(item["backtest"]["active_periods"]) for item in results]
  evaluation_rows = [int(item["evaluation_rows"]) for item in results]
  aggregate_metrics = [item.get("aggregate_metrics", {}) for item in results]
  gross_returns = [float(metrics.get("total_gross_return", 0.0)) for metrics in aggregate_metrics]
  transaction_costs = [float(metrics.get("total_transaction_cost", 0.0)) for metrics in aggregate_metrics]
  turnovers = [float(metrics.get("total_turnover", 0.0)) for metrics in aggregate_metrics]
  turnovers_per_period = [float(metrics.get("average_turnover_per_period", 0.0)) for metrics in aggregate_metrics]
  weak_primary = Counter(str(item["weakness"]["primary_weakness"]) for item in results)
  weak_secondary = Counter(str(item["weakness"]["secondary_weakness"]) for item in results)
  cost_share = [float(item["weakness"]["cost_share_of_gross"]) for item in results]
  resize_rate = [float(item["weakness"]["average_resize_rate"]) for item in results]
  metrics_keys = sorted({
    key
    for item in results
    for key, value in item.get("metrics", {}).items()
    if isinstance(value, (int, float))
  })

  metrics_summary = {
    key: _mean(float(item["metrics"][key]) for item in results if key in item.get("metrics", {}))
    for key in metrics_keys
  }
  compounded_net_return = math.prod(1.0 + value for value in net_returns) - 1.0
  total_gross_return = sum(gross_returns)
  total_transaction_cost = sum(transaction_costs)
  return {
    "model_name": model_name,
    "window_count": len(results),
    "positive_windows": sum(1 for value in net_returns if value > 0),
    "negative_windows": sum(1 for value in net_returns if value < 0),
    "flat_windows": sum(1 for value in net_returns if value == 0),
    "compounded_net_return": compounded_net_return,
    "mean_net_return": _mean(net_returns),
    "median_net_return": _median(net_returns),
    "mean_max_drawdown": _mean(max_drawdowns),
    "max_window_drawdown": max(max_drawdowns) if max_drawdowns else 0.0,
    "total_trades": sum(trades),
    "mean_trades": _mean(trades),
    "total_active_periods": sum(active_periods),
    "mean_active_periods": _mean(active_periods),
    "total_evaluation_rows": sum(evaluation_rows),
    "mean_evaluation_rows": _mean(evaluation_rows),
    "total_gross_return": total_gross_return,
    "total_transaction_cost": total_transaction_cost,
    "total_turnover": sum(turnovers),
    "mean_turnover_per_period": _mean(turnovers_per_period),
    "cost_share_of_gross_total": (total_transaction_cost / total_gross_return) if total_gross_return > 0 else 0.0,
    "mean_cost_share_of_gross": _mean(cost_share),
    "mean_resize_rate": _mean(resize_rate),
    "primary_weakness_counts": dict(weak_primary),
    "secondary_weakness_counts": dict(weak_secondary),
    "metrics_summary": metrics_summary,
  }


def _weakness_payload(summary: Any) -> dict[str, Any]:
  return {
    "overall_directional_accuracy": summary.overall_directional_accuracy,
    "trade_bar_directional_accuracy": summary.trade_bar_directional_accuracy,
    "active_bar_directional_accuracy": summary.active_bar_directional_accuracy,
    "average_hold_bars": summary.average_hold_bars,
    "average_resize_rate": summary.average_resize_rate,
    "cost_share_of_gross": summary.cost_share_of_gross,
    "turnover_cap_share": summary.turnover_cap_share,
    "primary_weakness": summary.primary_weakness,
    "secondary_weakness": summary.secondary_weakness,
  }


def _mean(values: Any) -> float:
  items = list(values)
  if not items:
    return 0.0
  return sum(float(value) for value in items) / len(items)


def _median(values: list[float]) -> float:
  if not values:
    return 0.0
  ordered = sorted(float(value) for value in values)
  midpoint = len(ordered) // 2
  if len(ordered) % 2 == 1:
    return ordered[midpoint]
  return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_dt(value: datetime) -> str:
  return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
