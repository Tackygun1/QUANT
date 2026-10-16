#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from scipy import stats


PALETTE = {
  "navy": "#17324d",
  "blue": "#2878b5",
  "teal": "#2a9d8f",
  "amber": "#e9a23b",
  "red": "#d1495b",
  "gray": "#67727e",
  "light_gray": "#d9e0e6",
  "paper": "#fbfaf7",
  "ink": "#18212b",
  "green": "#3a8f5b",
}


@dataclass(slots=True)
class BacktestRun:
  key: str
  label: str
  split: str
  policy: str
  root: Path
  summary: dict[str, Any]
  periods: list[dict[str, str]]
  trades: list[dict[str, str]]
  metrics: dict[str, Any]


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate a paper-style statistical report for the legacy hybrid direction model.")
  parser.add_argument("--config", type=Path, required=True)
  args = parser.parse_args()
  config = json.loads(args.config.read_text(encoding="utf-8"))
  output_dir = Path(config["output_dir"])
  figure_dir = output_dir / "figures"
  table_dir = output_dir / "tables"
  figure_dir.mkdir(parents=True, exist_ok=True)
  table_dir.mkdir(parents=True, exist_ok=True)

  run_specs = [
    ("validation_original", "Development · Original", "development_validation", "original", config["validation_original_backtest_dir"]),
    (
      "validation_low_turnover",
      "Development · Low-turnover",
      "development_validation",
      "low_turnover",
      config["validation_low_turnover_backtest_dir"],
    ),
    ("fresh_original", "Fresh OOS · Original", "fresh_out_of_sample", "original", config["fresh_original_backtest_dir"]),
    (
      "fresh_low_turnover",
      "Fresh OOS · Low-turnover",
      "fresh_out_of_sample",
      "low_turnover",
      config["fresh_low_turnover_backtest_dir"],
    ),
  ]
  runs = [
    load_run(
      key=key,
      label=label,
      split=split,
      policy=policy,
      root=Path(root),
      bar_minutes=int(config["bar_minutes"]),
      bootstrap_samples=int(config["bootstrap_samples"]),
      bootstrap_block_bars=int(config["bootstrap_block_bars"]),
      random_seed=int(config["random_seed"]) + index,
    )
    for index, (key, label, split, policy, root) in enumerate(run_specs)
  ]
  run_map = {run.key: run for run in runs}

  validation_predictions = load_predictions(Path(config["artifact_dir"]) / "validation_predictions.csv")
  fresh_predictions = load_predictions(Path(config["fresh_prediction_artifact_dir"]) / "validation_predictions.csv")
  classification = {
    "development_validation": classification_metrics(validation_predictions),
    "fresh_out_of_sample": classification_metrics(fresh_predictions),
  }
  walk_forward = json.loads(Path(config["walk_forward_summary_path"]).read_text(encoding="utf-8"))
  fee_grid = [float(value) for value in config["fee_sensitivity_bps"]]
  fee_sensitivity = {
    run.key: fixed_path_fee_sensitivity(run.periods, fee_grid)
    for run in runs
  }

  artifact_metadata = json.loads((Path(config["artifact_dir"]) / "metadata.json").read_text(encoding="utf-8"))
  report_payload = {
    "title": config["title"],
    "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "config_path": str(args.config),
    "artifact_dir": config["artifact_dir"],
    "artifact_metadata": artifact_metadata,
    "methodology": {
      "bar_minutes": int(config["bar_minutes"]),
      "bootstrap_samples": int(config["bootstrap_samples"]),
      "bootstrap_block_bars": int(config["bootstrap_block_bars"]),
      "bootstrap_method": "circular moving-block bootstrap of realized strategy bar returns",
      "fee_sensitivity_method": "fixed-position ex-post replay; signal path is held constant",
      "fresh_walk_forward_window_hours": 8,
    },
    "runs": {run.key: run.metrics for run in runs},
    "classification": classification,
    "fee_sensitivity": fee_sensitivity,
    "walk_forward": walk_forward,
    "research_decision": "Retain as a documented research result; do not promote for live deployment.",
  }
  (output_dir / "research_metrics.json").write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

  write_performance_table(table_dir / "performance_metrics.csv", runs)
  write_classification_table(table_dir / "classification_metrics.csv", classification)
  write_daily_table(table_dir / "daily_returns.csv", runs)
  write_walk_forward_table(table_dir / "walk_forward_windows.csv", walk_forward)
  write_fee_table(table_dir / "fee_sensitivity.csv", fee_sensitivity)

  make_equity_figure(figure_dir / "figure_01_equity_curves.svg", run_map)
  make_drawdown_figure(figure_dir / "figure_02_drawdown.svg", run_map)
  make_execution_figure(figure_dir / "figure_03_execution_costs.svg", runs)
  make_walk_forward_figure(figure_dir / "figure_04_walk_forward.svg", walk_forward)
  make_classification_figure(figure_dir / "figure_05_prediction_diagnostics.svg", classification)
  make_return_distribution_figure(figure_dir / "figure_06_return_distributions.svg", run_map)
  make_fee_sensitivity_figure(figure_dir / "figure_07_fee_sensitivity.svg", fee_sensitivity)
  make_exposure_figure(figure_dir / "figure_08_exposure.svg", run_map)

  markdown = build_markdown_report(config, runs, classification, walk_forward, fee_sensitivity, artifact_metadata)
  (output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
  (output_dir / "REPORT.html").write_text(build_html_report(markdown, config["title"]), encoding="utf-8")
  print(json.dumps({
    "output_dir": str(output_dir),
    "report": str(output_dir / "REPORT.html"),
    "figures": 8,
    "tables": 5,
    "decision": report_payload["research_decision"],
  }, indent=2))
  return 0


def load_run(
  *,
  key: str,
  label: str,
  split: str,
  policy: str,
  root: Path,
  bar_minutes: int,
  bootstrap_samples: int,
  bootstrap_block_bars: int,
  random_seed: int,
) -> BacktestRun:
  summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
  periods = read_csv(root / "periods.csv")
  trades = read_csv(root / "trades.csv")
  metrics = analyze_run(
    summary=summary,
    periods=periods,
    trades=trades,
    bar_minutes=bar_minutes,
    bootstrap_samples=bootstrap_samples,
    bootstrap_block_bars=bootstrap_block_bars,
    random_seed=random_seed,
  )
  return BacktestRun(
    key=key,
    label=label,
    split=split,
    policy=policy,
    root=root,
    summary=summary,
    periods=periods,
    trades=trades,
    metrics=metrics,
  )


def analyze_run(
  *,
  summary: dict[str, Any],
  periods: list[dict[str, str]],
  trades: list[dict[str, str]],
  bar_minutes: int,
  bootstrap_samples: int,
  bootstrap_block_bars: int,
  random_seed: int,
) -> dict[str, Any]:
  net = values(periods, "net_return")
  gross = values(periods, "gross_return")
  realized = values(periods, "realized_return")
  positions = values(periods, "target_position")
  equity = cumulative_equity(net)
  buy_hold_equity = cumulative_equity(realized)
  drawdowns = drawdown_series(equity)
  daily = daily_compounded_returns(periods)
  bootstrap_low, bootstrap_high, bootstrap_probability_positive = block_bootstrap_total_return(
    net,
    samples=bootstrap_samples,
    block_size=min(bootstrap_block_bars, len(net)),
    seed=random_seed,
  )
  mean_return = float(np.mean(net)) if net else 0.0
  std_return = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
  downside = np.asarray([value for value in net if value < 0], dtype=float)
  periods_per_year = (60 / bar_minutes) * 24 * 365
  sortino = (
    mean_return / float(np.std(downside, ddof=1)) * math.sqrt(periods_per_year)
    if len(downside) > 1 and float(np.std(downside, ddof=1)) > 0
    else 0.0
  )
  q05 = float(np.quantile(net, 0.05)) if net else 0.0
  cvar = float(np.mean([value for value in net if value <= q05])) if net else 0.0
  start = parse_dt(periods[0]["candle_start"])
  end = parse_dt(periods[-1]["candle_start"])
  annualized_volatility = std_return * math.sqrt(periods_per_year)
  long_share = sum(position > 1e-12 for position in positions) / len(positions)
  short_share = sum(position < -1e-12 for position in positions) / len(positions)
  flat_share = 1.0 - long_share - short_share
  active_abs = [abs(position) for position in positions if abs(position) > 1e-12]
  aggregate = summary.get("aggregate_metrics", {})
  entry_count = int(float(aggregate.get("entry_count", 0.0)))
  flip_count = int(float(aggregate.get("flip_count", 0.0)))
  trade_turnover = values(trades, "turnover")
  hac_t, hac_se = newey_west_mean_t(net, lag=min(12, max(0, len(net) - 1)))
  return {
    "start_time": start.isoformat().replace("+00:00", "Z"),
    "end_time": end.isoformat().replace("+00:00", "Z"),
    "calendar_days": (end - start).total_seconds() / 86400,
    "bars": len(periods),
    "gross_return": float(summary.get("aggregate_metrics", {}).get("total_gross_return", sum(gross))),
    "net_return": float(summary["total_net_return"]),
    "buy_and_hold_return": buy_hold_equity[-1] - 1.0,
    "transaction_cost": float(summary["total_transaction_cost"]),
    "cost_share_of_positive_gross": (
      float(summary["total_transaction_cost"]) / float(summary["aggregate_metrics"]["total_gross_return"])
      if float(summary["aggregate_metrics"]["total_gross_return"]) > 0
      else None
    ),
    "turnover": float(summary["total_turnover"]),
    "position_adjustments": int(summary["trades"]),
    "entries": entry_count,
    "flips": flip_count,
    "active_periods": int(summary["active_periods"]),
    "active_share": int(summary["active_periods"]) / len(periods),
    "active_hit_rate": float(summary["active_hit_rate"]),
    "profit_factor_after_cost": float(summary["profit_factor_after_cost"]),
    "max_drawdown": float(summary["max_drawdown"]),
    "max_drawdown_duration_bars": max_drawdown_duration(equity),
    "max_drawdown_duration_hours": max_drawdown_duration(equity) * bar_minutes / 60,
    "sharpe_like_annualized": float(summary["sharpe_like"]),
    "sortino_like_annualized": sortino,
    "annualized_volatility": annualized_volatility,
    "mean_bar_return": mean_return,
    "bar_return_std": std_return,
    "bar_return_skew": float(stats.skew(net, bias=False)) if len(net) > 2 else 0.0,
    "bar_return_excess_kurtosis": float(stats.kurtosis(net, fisher=True, bias=False)) if len(net) > 3 else 0.0,
    "bar_var_95": -q05,
    "bar_cvar_95": -cvar,
    "lag_1_return_autocorrelation": lag_correlation(net, 1),
    "newey_west_mean_return_t_stat": hac_t,
    "newey_west_mean_return_standard_error": hac_se,
    "bootstrap_total_return_ci_95": [bootstrap_low, bootstrap_high],
    "bootstrap_probability_positive": bootstrap_probability_positive,
    "positive_days": sum(value > 0 for value in daily.values()),
    "daily_observations": len(daily),
    "positive_day_share": sum(value > 0 for value in daily.values()) / len(daily) if daily else 0.0,
    "best_day": max(daily.values()) if daily else 0.0,
    "worst_day": min(daily.values()) if daily else 0.0,
    "mean_abs_exposure": float(np.mean(np.abs(positions))),
    "mean_active_abs_exposure": float(np.mean(active_abs)) if active_abs else 0.0,
    "long_exposure_share": long_share,
    "short_exposure_share": short_share,
    "flat_share": flat_share,
    "median_adjustment_turnover": median(trade_turnover) if trade_turnover else 0.0,
    "p95_adjustment_turnover": float(np.quantile(trade_turnover, 0.95)) if trade_turnover else 0.0,
    "break_even_total_cost_bps_fixed_path": break_even_cost_bps(periods),
    "ending_equity": equity[-1],
    "buy_and_hold_ending_equity": buy_hold_equity[-1],
    "maximum_observed_drawdown_from_series": max(drawdowns) if drawdowns else 0.0,
  }


def classification_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
  classes = (-1, 0, 1)
  targets = [int(float(row["target"])) for row in rows]
  predictions = [int(float(row["prediction"])) for row in rows]
  confusion = [[sum(t == actual and p == predicted for t, p in zip(targets, predictions)) for predicted in classes] for actual in classes]
  per_class: dict[str, dict[str, float | int]] = {}
  precisions: list[float] = []
  recalls: list[float] = []
  f1s: list[float] = []
  for index, cls in enumerate(classes):
    tp = confusion[index][index]
    actual_total = sum(confusion[index])
    predicted_total = sum(row[index] for row in confusion)
    precision = tp / predicted_total if predicted_total else 0.0
    recall = tp / actual_total if actual_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    per_class[str(cls)] = {
      "support": actual_total,
      "predicted": predicted_total,
      "precision": precision,
      "recall": recall,
      "f1": f1,
    }
  n = len(rows)
  correct = sum(target == prediction for target, prediction in zip(targets, predictions))
  actual_counts = Counter(targets)
  predicted_counts = Counter(predictions)
  accuracy = correct / n
  expected_accuracy = sum(actual_counts[cls] * predicted_counts[cls] for cls in classes) / (n * n)
  kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy) if expected_accuracy < 1 else 0.0
  c = correct
  sum_pk_tk = sum(predicted_counts[cls] * actual_counts[cls] for cls in classes)
  denominator = math.sqrt(
    (n * n - sum(predicted_counts[cls] ** 2 for cls in classes))
    * (n * n - sum(actual_counts[cls] ** 2 for cls in classes))
  )
  mcc = (c * n - sum_pk_tk) / denominator if denominator else 0.0
  return {
    "rows": n,
    "accuracy": accuracy,
    "balanced_accuracy": sum(recalls) / len(recalls),
    "macro_precision": sum(precisions) / len(precisions),
    "macro_recall": sum(recalls) / len(recalls),
    "macro_f1": sum(f1s) / len(f1s),
    "cohen_kappa": kappa,
    "matthews_correlation": mcc,
    "dominant_prediction_share": max(predicted_counts.values()) / n,
    "target_distribution": {str(cls): actual_counts[cls] for cls in classes},
    "prediction_distribution": {str(cls): predicted_counts[cls] for cls in classes},
    "confusion_matrix": confusion,
    "class_order": list(classes),
    "per_class": per_class,
  }


def block_bootstrap_total_return(
  returns: list[float],
  *,
  samples: int,
  block_size: int,
  seed: int,
) -> tuple[float, float, float]:
  if not returns or samples <= 0:
    return 0.0, 0.0, 0.0
  data = np.asarray(returns, dtype=float)
  n = len(data)
  block_size = max(1, min(block_size, n))
  blocks_needed = math.ceil(n / block_size)
  rng = np.random.default_rng(seed)
  totals = np.empty(samples, dtype=float)
  offsets = np.arange(block_size)
  for sample_index in range(samples):
    starts = rng.integers(0, n, size=blocks_needed)
    indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
    sampled = data[indices]
    totals[sample_index] = float(np.expm1(np.log1p(np.maximum(sampled, -0.999999)).sum()))
  return (
    float(np.quantile(totals, 0.025)),
    float(np.quantile(totals, 0.975)),
    float(np.mean(totals > 0)),
  )


def newey_west_mean_t(values_: list[float], lag: int) -> tuple[float, float]:
  if len(values_) < 2:
    return 0.0, 0.0
  data = np.asarray(values_, dtype=float)
  centered = data - data.mean()
  n = len(data)
  long_run_variance = float(centered @ centered / n)
  for current_lag in range(1, lag + 1):
    covariance = float(centered[current_lag:] @ centered[:-current_lag] / n)
    long_run_variance += 2 * (1 - current_lag / (lag + 1)) * covariance
  standard_error = math.sqrt(max(0.0, long_run_variance) / n)
  return (float(data.mean() / standard_error) if standard_error > 0 else 0.0, standard_error)


def fixed_path_fee_sensitivity(periods: list[dict[str, str]], fee_grid: list[float]) -> list[dict[str, float]]:
  gross = values(periods, "gross_return")
  turnover = values(periods, "turnover")
  results = []
  for total_cost_bps in fee_grid:
    returns = [gross_return - traded * total_cost_bps / 10_000 for gross_return, traded in zip(gross, turnover)]
    results.append({
      "total_cost_bps": total_cost_bps,
      "net_return": compounded_return(returns),
      "transaction_cost_sum": sum(traded * total_cost_bps / 10_000 for traded in turnover),
    })
  return results


def break_even_cost_bps(periods: list[dict[str, str]]) -> float:
  gross = values(periods, "gross_return")
  turnover = values(periods, "turnover")
  if compounded_return(gross) <= 0 or sum(turnover) <= 0:
    return 0.0
  low = 0.0
  high = 100.0
  for _ in range(80):
    midpoint = (low + high) / 2
    result = compounded_return([value - traded * midpoint / 10_000 for value, traded in zip(gross, turnover)])
    if result > 0:
      low = midpoint
    else:
      high = midpoint
  return (low + high) / 2


def write_performance_table(path: Path, runs: list[BacktestRun]) -> None:
  fields = [
    "run",
    "split",
    "policy",
    "start_time",
    "end_time",
    "calendar_days",
    "bars",
    "gross_return",
    "net_return",
    "buy_and_hold_return",
    "transaction_cost",
    "turnover",
    "position_adjustments",
    "entries",
    "active_share",
    "active_hit_rate",
    "max_drawdown",
    "sharpe_like_annualized",
    "sortino_like_annualized",
    "profit_factor_after_cost",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_probability_positive",
    "newey_west_mean_return_t_stat",
    "break_even_total_cost_bps_fixed_path",
  ]
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for run in runs:
      metrics = run.metrics
      row = {field: metrics.get(field, "") for field in fields}
      row.update({
        "run": run.key,
        "split": run.split,
        "policy": run.policy,
        "bootstrap_ci_low": metrics["bootstrap_total_return_ci_95"][0],
        "bootstrap_ci_high": metrics["bootstrap_total_return_ci_95"][1],
      })
      writer.writerow(row)


def write_classification_table(path: Path, classification: dict[str, dict[str, Any]]) -> None:
  fields = [
    "split",
    "rows",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "cohen_kappa",
    "matthews_correlation",
    "dominant_prediction_share",
  ]
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for split, metrics in classification.items():
      writer.writerow({"split": split, **{field: metrics.get(field, "") for field in fields if field != "split"}})


def write_daily_table(path: Path, runs: list[BacktestRun]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["run", "date", "net_return"])
    writer.writeheader()
    for run in runs:
      for date, result in daily_compounded_returns(run.periods).items():
        writer.writerow({"run": run.key, "date": date, "net_return": result})


def write_walk_forward_table(path: Path, walk_forward: dict[str, Any]) -> None:
  fields = [
    "policy",
    "index",
    "start_time",
    "end_time",
    "rows",
    "accuracy",
    "gross_return",
    "total_cost",
    "net_return",
    "total_turnover",
    "trade_adjustments",
    "entries",
    "max_drawdown",
  ]
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for policy in ("original", "low_turnover"):
      for window in walk_forward["policies"][policy]["windows"]:
        writer.writerow({"policy": policy, **{field: window.get(field, "") for field in fields if field != "policy"}})


def write_fee_table(path: Path, fee_sensitivity: dict[str, list[dict[str, float]]]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["run", "total_cost_bps", "net_return", "transaction_cost_sum"])
    writer.writeheader()
    for run, rows in fee_sensitivity.items():
      for row in rows:
        writer.writerow({"run": run, **row})


def make_equity_figure(path: Path, run_map: dict[str, BacktestRun]) -> None:
  elements = svg_canvas("Figure 1. Strategy and benchmark equity", "Compounded growth of $1; all series begin at 1.00")
  for panel_index, (title, original_key, low_key) in enumerate([
    ("Development validation", "validation_original", "validation_low_turnover"),
    ("Fresh out-of-sample", "fresh_original", "fresh_low_turnover"),
  ]):
    original = run_map[original_key]
    low = run_map[low_key]
    series = [
      ("Original", cumulative_equity(values(original.periods, "net_return")), PALETTE["blue"]),
      ("Low-turnover", cumulative_equity(values(low.periods, "net_return")), PALETTE["teal"]),
      ("BTC buy-and-hold", cumulative_equity(values(original.periods, "realized_return")), PALETTE["gray"]),
    ]
    line_panel(elements, 70 + panel_index * 580, 120, 520, 430, series, title, percent_axis=False)
  add_legend(elements, 360, 610, [("Original", PALETTE["blue"]), ("Low-turnover", PALETTE["teal"]), ("BTC buy-and-hold", PALETTE["gray"])])
  save_svg(path, elements)


def make_drawdown_figure(path: Path, run_map: dict[str, BacktestRun]) -> None:
  elements = svg_canvas("Figure 2. Drawdown paths", "Peak-to-trough loss through time; lower is worse")
  for panel_index, (title, original_key, low_key) in enumerate([
    ("Development validation", "validation_original", "validation_low_turnover"),
    ("Fresh out-of-sample", "fresh_original", "fresh_low_turnover"),
  ]):
    original = run_map[original_key]
    low = run_map[low_key]
    series = [
      ("Original", [-value for value in drawdown_series(cumulative_equity(values(original.periods, "net_return")))], PALETTE["blue"]),
      ("Low-turnover", [-value for value in drawdown_series(cumulative_equity(values(low.periods, "net_return")))], PALETTE["teal"]),
    ]
    line_panel(elements, 70 + panel_index * 580, 120, 520, 430, series, title, percent_axis=True, include_zero=True)
  add_legend(elements, 430, 610, [("Original", PALETTE["blue"]), ("Low-turnover", PALETTE["teal"])])
  save_svg(path, elements)


def make_execution_figure(path: Path, runs: list[BacktestRun]) -> None:
  elements = svg_canvas("Figure 3. Execution intensity and implementation costs", "Position adjustments are not round trips; cost assumption is 1.0 bp fee + 0.5 bp slippage")
  labels = ["Dev original", "Dev low-turn", "OOS original", "OOS low-turn"]
  colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["blue"], PALETTE["teal"]]
  bar_panel(elements, 55, 135, 350, 400, labels, [run.metrics["position_adjustments"] for run in runs], colors, "Position adjustments", "integer")
  bar_panel(elements, 425, 135, 350, 400, labels, [run.metrics["turnover"] for run in runs], colors, "Total turnover", "decimal")
  bar_panel(elements, 795, 135, 350, 400, labels, [run.metrics["transaction_cost"] for run in runs], colors, "Modeled cost", "percent")
  add_legend(elements, 430, 610, [("Original execution", PALETTE["blue"]), ("Low-turnover execution", PALETTE["teal"])])
  save_svg(path, elements)


def make_walk_forward_figure(path: Path, walk_forward: dict[str, Any]) -> None:
  elements = svg_canvas("Figure 4. Fresh rolling walk-forward returns", "Independent 8-hour windows; window 6 is a 3.6-hour partial tail")
  original = [float(row["net_return"]) for row in walk_forward["policies"]["original"]["windows"]]
  low = [float(row["net_return"]) for row in walk_forward["policies"]["low_turnover"]["windows"]]
  grouped_bar_panel(
    elements,
    90,
    130,
    1020,
    430,
    [f"W{index}" for index in range(1, len(original) + 1)],
    [("Original", original, PALETTE["blue"]), ("Low-turnover", low, PALETTE["teal"])],
    "Window net return",
    "percent",
  )
  add_legend(elements, 430, 620, [("Original", PALETTE["blue"]), ("Low-turnover", PALETTE["teal"])])
  save_svg(path, elements)


def make_classification_figure(path: Path, classification: dict[str, dict[str, Any]]) -> None:
  elements = svg_canvas("Figure 5. Predictive behavior and regime collapse", "Class labels: −1 down, 0 neutral, +1 up")
  classes = ["−1", "0", "+1"]
  development = classification["development_validation"]
  fresh = classification["fresh_out_of_sample"]
  grouped_bar_panel(
    elements,
    60,
    140,
    530,
    390,
    classes,
    [
      ("Development predictions", [development["prediction_distribution"][str(cls)] / development["rows"] for cls in (-1, 0, 1)], PALETTE["blue"]),
      ("Fresh predictions", [fresh["prediction_distribution"][str(cls)] / fresh["rows"] for cls in (-1, 0, 1)], PALETTE["red"]),
    ],
    "Prediction class share",
    "percent",
  )
  confusion_panel(elements, 680, 150, 390, fresh["confusion_matrix"], classes, "Fresh OOS confusion matrix")
  add_legend(elements, 310, 620, [("Development predictions", PALETTE["blue"]), ("Fresh predictions", PALETTE["red"])])
  save_svg(path, elements)


def make_return_distribution_figure(path: Path, run_map: dict[str, BacktestRun]) -> None:
  elements = svg_canvas("Figure 6. Distribution of realized strategy returns", "Histogram density of five-minute net returns; tails are clipped to the joint 0.5–99.5% display range")
  for panel_index, (title, original_key, low_key) in enumerate([
    ("Development validation", "validation_original", "validation_low_turnover"),
    ("Fresh out-of-sample", "fresh_original", "fresh_low_turnover"),
  ]):
    original = values(run_map[original_key].periods, "net_return")
    low = values(run_map[low_key].periods, "net_return")
    histogram_panel(
      elements,
      65 + panel_index * 575,
      130,
      500,
      410,
      [("Original", original, PALETTE["blue"]), ("Low-turnover", low, PALETTE["teal"])],
      title,
    )
  add_legend(elements, 430, 620, [("Original", PALETTE["blue"]), ("Low-turnover", PALETTE["teal"])])
  save_svg(path, elements)


def make_fee_sensitivity_figure(path: Path, fee_sensitivity: dict[str, list[dict[str, float]]]) -> None:
  elements = svg_canvas("Figure 7. Fixed-path transaction-cost sensitivity", "Combined fee + slippage; positions are held fixed, so this is an implementation shortfall stress test")
  specs = [
    ("Development validation", "validation_original", "validation_low_turnover"),
    ("Fresh out-of-sample", "fresh_original", "fresh_low_turnover"),
  ]
  for panel_index, (title, original_key, low_key) in enumerate(specs):
    x_values = [row["total_cost_bps"] for row in fee_sensitivity[original_key]]
    series = [
      ("Original", [row["net_return"] for row in fee_sensitivity[original_key]], PALETTE["blue"]),
      ("Low-turnover", [row["net_return"] for row in fee_sensitivity[low_key]], PALETTE["teal"]),
    ]
    line_panel(elements, 70 + panel_index * 580, 120, 520, 430, series, title, percent_axis=True, include_zero=True, x_values=x_values, x_label="total cost (bps)")
  add_legend(elements, 430, 620, [("Original", PALETTE["blue"]), ("Low-turnover", PALETTE["teal"])])
  save_svg(path, elements)


def make_exposure_figure(path: Path, run_map: dict[str, BacktestRun]) -> None:
  elements = svg_canvas("Figure 8. Position exposure through time", "Target position after sizing and risk constraints; positive is long and negative is short")
  for panel_index, (title, original_key, low_key) in enumerate([
    ("Development validation", "validation_original", "validation_low_turnover"),
    ("Fresh out-of-sample", "fresh_original", "fresh_low_turnover"),
  ]):
    series = [
      ("Original", values(run_map[original_key].periods, "target_position"), PALETTE["blue"]),
      ("Low-turnover", values(run_map[low_key].periods, "target_position"), PALETTE["teal"]),
    ]
    line_panel(elements, 70 + panel_index * 580, 120, 520, 430, series, title, percent_axis=False, include_zero=True, y_fixed=(-1.05, 1.05))
  add_legend(elements, 430, 620, [("Original", PALETTE["blue"]), ("Low-turnover", PALETTE["teal"])])
  save_svg(path, elements)


def build_markdown_report(
  config: dict[str, Any],
  runs: list[BacktestRun],
  classification: dict[str, dict[str, Any]],
  walk_forward: dict[str, Any],
  fee_sensitivity: dict[str, list[dict[str, float]]],
  artifact_metadata: dict[str, Any],
) -> str:
  run_map = {run.key: run for run in runs}
  dev_original = run_map["validation_original"].metrics
  dev_low = run_map["validation_low_turnover"].metrics
  oos_original = run_map["fresh_original"].metrics
  oos_low = run_map["fresh_low_turnover"].metrics
  dev_class = classification["development_validation"]
  oos_class = classification["fresh_out_of_sample"]
  full_original = walk_forward["policies"]["original"]["full_windows_only"]
  full_low = walk_forward["policies"]["low_turnover"]["full_windows_only"]
  trade_reduction = 1 - full_low["trade_adjustments"] / full_original["trade_adjustments"]
  cost_reduction = 1 - full_low["total_cost"] / full_original["total_cost"]
  rows = [
    "Development / original",
    "Development / low-turnover",
    "Fresh OOS / original",
    "Fresh OOS / low-turnover",
  ]
  metrics_rows = [dev_original, dev_low, oos_original, oos_low]
  table_lines = [
    "| Sample and execution | Gross return | Net return | Costs | Adjustments | Turnover | Max DD | Sharpe-like |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for label, metric in zip(rows, metrics_rows):
    table_lines.append(
      f"| {label} | {pct(metric['gross_return'])} | {pct(metric['net_return'])} | {pct(metric['transaction_cost'])} | "
      f"{metric['position_adjustments']:,} | {metric['turnover']:.2f} | {pct(metric['max_drawdown'])} | {metric['sharpe_like_annualized']:.2f} |"
    )
  report = f"""# {config['title']}

## Abstract

This report audits the saved `lightgbm_hybrid_direction_meta` artifact, a LightGBM direction classifier using market features and stacked hybrid-transformer forecasts. On its development validation interval, the low-turnover execution policy improved net return from {pct(dev_original['net_return'])} to {pct(dev_low['net_return'])} while reducing modeled transaction costs from {pct(dev_original['transaction_cost'])} to {pct(dev_low['transaction_cost'])}. The result did not generalize: on the later frozen sample, both policies lost approximately 3.2%. Fresh predictions collapsed to the short class on {pct(oos_class['dominant_prediction_share'])} of observations while BTC appreciated {pct(oos_original['buy_and_hold_return'])}. The artifact is valuable as a documented research result but is not supported for live promotion.

## 1. Experimental design

- **Instrument and frequency:** Coinbase BTC-USD, five-minute bars.
- **Model:** `{artifact_metadata['model_name']}`; target `{artifact_metadata['target_column']}`.
- **Development validation:** {dev_original['bars']:,} observations from {dev_original['start_time']} through {dev_original['end_time']} ({dev_original['calendar_days']:.1f} calendar days). This interval was used for execution-policy selection and is therefore not an untouched test.
- **Fresh out-of-sample:** {oos_original['bars']:,} observations from {oos_original['start_time']} through {oos_original['end_time']} ({oos_original['calendar_days']:.1f} calendar days).
- **Implementation costs:** 1.0 bp fee plus 0.5 bp slippage per unit of turnover.
- **Robustness:** five complete non-overlapping eight-hour walk-forward windows plus one partial tail window, resetting execution state at each boundary.
- **Uncertainty:** 5,000-replicate circular moving-block bootstrap with 24-bar blocks. This quantifies path uncertainty but cannot compensate for the short calendar span.

## 2. Headline performance

{chr(10).join(table_lines)}

The development result is economically attractive but statistically fragile. The low-turnover development return has a 95% moving-block bootstrap interval of {pct(dev_low['bootstrap_total_return_ci_95'][0])} to {pct(dev_low['bootstrap_total_return_ci_95'][1])}. The fresh low-turnover interval is {pct(oos_low['bootstrap_total_return_ci_95'][0])} to {pct(oos_low['bootstrap_total_return_ci_95'][1])}. Annualized Sharpe-like values are shown for comparability, but their magnitude is not reliable over samples lasting only days.

![Equity curves](figures/figure_01_equity_curves.svg)

![Drawdown paths](figures/figure_02_drawdown.svg)

## 3. Execution and transaction-cost ablation

On development validation, low-turnover execution reduced position adjustments from {dev_original['position_adjustments']:,} to {dev_low['position_adjustments']:,} and turnover from {dev_original['turnover']:.2f} to {dev_low['turnover']:.2f}. Its fixed-path break-even combined cost was {dev_low['break_even_total_cost_bps_fixed_path']:.2f} bps versus {dev_original['break_even_total_cost_bps_fixed_path']:.2f} bps for the original policy.

On the five complete fresh walk-forward windows, adjustments fell {pct(trade_reduction)} and modeled cost fell {pct(cost_reduction)}. Net return improved from {pct(full_original['compounded_net_return'])} to {pct(full_low['compounded_net_return'])}, but only {full_low['positive_windows']} of {full_low['window_count']} low-turnover windows were profitable and aggregate profitability remained negative.

![Execution costs](figures/figure_03_execution_costs.svg)

![Walk-forward windows](figures/figure_04_walk_forward.svg)

![Fee sensitivity](figures/figure_07_fee_sensitivity.svg)

## 4. Predictive diagnostics

Development accuracy was {pct(dev_class['accuracy'])}, balanced accuracy was {pct(dev_class['balanced_accuracy'])}, and multiclass Matthews correlation was {dev_class['matthews_correlation']:.3f}. Fresh accuracy fell to {pct(oos_class['accuracy'])}, balanced accuracy to {pct(oos_class['balanced_accuracy'])}, and Matthews correlation to {oos_class['matthews_correlation']:.3f}.

The decisive failure is distributional: {oos_class['prediction_distribution']['-1']} of {oos_class['rows']} fresh predictions were short. Actual fresh labels were materially balanced across down, neutral, and up classes. This is consistent with feature or calibration drift in the meta-model rather than a broker-cost problem alone.

![Prediction diagnostics](figures/figure_05_prediction_diagnostics.svg)

## 5. Risk, tails, and exposure

The development low-turnover maximum drawdown was {pct(dev_low['max_drawdown'])}, versus {pct(dev_original['max_drawdown'])} originally. Fresh maximum drawdown exceeded 4.2% for both policies. Development low-turnover five-minute 95% historical VaR and expected shortfall were {pct(dev_low['bar_var_95'])} and {pct(dev_low['bar_cvar_95'])}; fresh values were {pct(oos_low['bar_var_95'])} and {pct(oos_low['bar_cvar_95'])}.

Fresh low-turnover exposure was short {pct(oos_low['short_exposure_share'])} of bars and long {pct(oos_low['long_exposure_share'])}. This persistent directional exposure explains why lower order count did not rescue performance when the predictive distribution shifted.

![Return distributions](figures/figure_06_return_distributions.svg)

![Position exposure](figures/figure_08_exposure.svg)

## 6. Statistical and methodological limitations

1. The development interval spans only {dev_original['calendar_days']:.1f} calendar days and the fresh interval only {oos_original['calendar_days']:.1f} days. Neither supports a stable estimate of annual performance.
2. Execution parameters were chosen using the development validation interval. Its performance is selection-biased and must not be described as final out-of-sample evidence.
3. The fresh sample contains only two calendar dates and one market regime. It is sufficient to reject immediate deployment, not to prove the model has no possible edge.
4. Backtests use a proportional turnover cost model. Exchange tiering, spread variation, order-book impact, rejected orders, latency, funding, and taxes are outside scope.
5. Fee sensitivity holds the realized position path fixed. It is an implementation-shortfall stress test, not a full re-simulation under changing risk constraints.
6. Position-adjustment counts are order events, not completed round-trip trades.

## 7. Conclusion

The low-turnover overlay successfully reduces order frequency and improves implementation efficiency in rolling evaluation. It does not preserve profitability on later data because the saved classifier's direction distribution collapses. The defensible project conclusion is therefore **retain for research documentation, reject for live deployment**. Any future retrain should be pre-registered, explicitly monitor prediction-class drift, calibrate probabilities or margins, and reserve a new untouched holdout after all model and execution choices are frozen.

## Reproducibility appendix

- Configuration: `{config['output_dir']}` was generated from `configs/research/runtime/old_hybrid_direction_research_report_20260816.json`.
- Machine-readable metrics: `research_metrics.json`.
- Tabular exports: `tables/performance_metrics.csv`, `tables/classification_metrics.csv`, `tables/daily_returns.csv`, `tables/walk_forward_windows.csv`, and `tables/fee_sensitivity.csv`.
- Figures are standalone SVG vector graphics suitable for browser display or import into a typesetting system.
"""
  return report


def build_html_report(markdown: str, title: str) -> str:
  lines = markdown.splitlines()
  body: list[str] = []
  in_list = False
  in_table = False
  table_rows: list[list[str]] = []

  def close_list() -> None:
    nonlocal in_list
    if in_list:
      body.append("</ol>")
      in_list = False

  def flush_table() -> None:
    nonlocal in_table, table_rows
    if not in_table:
      return
    headers = table_rows[0]
    body.append("<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{cell}</th>" for cell in headers) + "</tr></thead><tbody>")
    for row in table_rows[2:]:
      body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    body.append("</tbody></table></div>")
    in_table = False
    table_rows = []

  for raw in lines:
    line = raw.strip()
    if line.startswith("|") and line.endswith("|"):
      close_list()
      in_table = True
      table_rows.append([inline_markup(cell.strip()) for cell in line.strip("|").split("|")])
      continue
    flush_table()
    if not line:
      close_list()
      continue
    if line.startswith("# "):
      close_list()
      body.append(f"<h1>{inline_markup(line[2:])}</h1>")
    elif line.startswith("## "):
      close_list()
      body.append(f"<h2>{inline_markup(line[3:])}</h2>")
    elif line.startswith("![") and "](" in line and line.endswith(")"):
      close_list()
      alt, source = line[2:].split("](", 1)
      body.append(f"<figure><img src='{html.escape(source[:-1])}' alt='{html.escape(alt)}'><figcaption>{html.escape(alt)}</figcaption></figure>")
    elif line[0].isdigit() and ". " in line:
      if not in_list:
        body.append("<ol>")
        in_list = True
      body.append(f"<li>{inline_markup(line.split('. ', 1)[1])}</li>")
    elif line.startswith("- "):
      close_list()
      body.append(f"<p class='bullet'>• {inline_markup(line[2:])}</p>")
    else:
      close_list()
      body.append(f"<p>{inline_markup(line)}</p>")
  flush_table()
  close_list()
  return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ --ink: #18212b; --muted: #5b6672; --rule: #d8dee5; --paper: #fbfaf7; --accent: #17324d; }}
  body {{ margin: 0; background: #e9edf0; color: var(--ink); font: 16px/1.58 Georgia, 'Times New Roman', serif; }}
  article {{ max-width: 1040px; margin: 32px auto; padding: 64px 78px; background: var(--paper); box-shadow: 0 3px 22px #1a273322; }}
  h1 {{ color: var(--accent); font-size: 2.35rem; line-height: 1.15; margin: 0 0 2rem; border-bottom: 3px solid var(--accent); padding-bottom: 1rem; }}
  h2 {{ color: var(--accent); font: 700 1.55rem/1.25 Arial, sans-serif; margin-top: 2.8rem; border-bottom: 1px solid var(--rule); padding-bottom: .35rem; }}
  p, li {{ max-width: 88ch; }} code {{ background: #edf1f3; padding: .08rem .28rem; border-radius: 3px; }}
  figure {{ margin: 2.2rem 0 2.8rem; }} img {{ width: 100%; height: auto; border: 1px solid var(--rule); background: white; }}
  figcaption {{ color: var(--muted); text-align: center; font-style: italic; font-size: .9rem; margin-top: .45rem; }}
  .table-wrap {{ overflow-x: auto; margin: 1.5rem 0 2rem; }} table {{ border-collapse: collapse; width: 100%; font: .86rem/1.35 Arial, sans-serif; }}
  th {{ background: var(--accent); color: white; text-align: left; }} th, td {{ padding: .62rem .58rem; border: 1px solid var(--rule); }} tbody tr:nth-child(even) {{ background: #f0f3f4; }}
  .bullet {{ margin: .28rem 0 .28rem 1rem; }} strong {{ color: #132b43; }}
  @media (max-width: 800px) {{ article {{ margin: 0; padding: 30px 20px; }} h1 {{ font-size: 1.8rem; }} }}
  @media print {{ body {{ background: white; }} article {{ margin: 0; max-width: none; box-shadow: none; }} figure {{ break-inside: avoid; }} }}
</style>
</head>
<body><article>{''.join(body)}</article></body>
</html>
"""


def inline_markup(value: str) -> str:
  escaped = html.escape(value)
  parts = escaped.split("**")
  if len(parts) > 1:
    escaped = "".join(f"<strong>{part}</strong>" if index % 2 else part for index, part in enumerate(parts))
  parts = escaped.split("`")
  if len(parts) > 1:
    escaped = "".join(f"<code>{part}</code>" if index % 2 else part for index, part in enumerate(parts))
  return escaped


def svg_canvas(title: str, subtitle: str) -> list[str]:
  return [
    "<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='680' viewBox='0 0 1200 680' role='img'>",
    f"<rect width='1200' height='680' fill='{PALETTE['paper']}'/>",
    f"<text x='60' y='52' font-family='Arial,sans-serif' font-size='25' font-weight='700' fill='{PALETTE['ink']}'>{html.escape(title)}</text>",
    f"<text x='60' y='82' font-family='Arial,sans-serif' font-size='14' fill='{PALETTE['gray']}'>{html.escape(subtitle)}</text>",
  ]


def save_svg(path: Path, elements: list[str]) -> None:
  path.write_text("\n".join(elements + ["</svg>"]), encoding="utf-8")


def line_panel(
  elements: list[str],
  x: float,
  y: float,
  width: float,
  height: float,
  series: list[tuple[str, list[float], str]],
  title: str,
  *,
  percent_axis: bool,
  include_zero: bool = False,
  x_values: list[float] | None = None,
  x_label: str = "elapsed sample",
  y_fixed: tuple[float, float] | None = None,
) -> None:
  all_values = [value for _, values_, _ in series for value in values_ if math.isfinite(value)]
  y_min, y_max = y_fixed or padded_range(all_values, include_zero=include_zero)
  elements.append(f"<text x='{x}' y='{y - 18}' font-family='Arial,sans-serif' font-size='17' font-weight='700' fill='{PALETTE['navy']}'>{html.escape(title)}</text>")
  draw_axes(elements, x, y, width, height, y_min, y_max, percent_axis=percent_axis)
  for _, values_, color in series:
    if not values_:
      continue
    points = []
    for index, value in enumerate(values_):
      if x_values is not None and len(x_values) == len(values_):
        x_min = min(x_values)
        x_max = max(x_values)
        px = x + (x_values[index] - x_min) / max(1e-12, x_max - x_min) * width
      else:
        px = x + (index / max(1, len(values_) - 1)) * width
      py = y + height - (value - y_min) / max(1e-12, y_max - y_min) * height
      points.append(f"{px:.2f},{py:.2f}")
    elements.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2.1' stroke-linejoin='round' opacity='.94'/>")
  if x_values is not None:
    x_min = min(x_values)
    x_max = max(x_values)
    for tick in (x_min, (x_min + x_max) / 2, x_max):
      px = x + (tick - x_min) / max(1e-12, x_max - x_min) * width
      label = f"{tick:g}"
      elements.append(f"<text x='{px:.1f}' y='{y + height + 22}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>{label}</text>")
  else:
    for tick in (0.0, 0.5, 1.0):
      px = x + tick * width
      label = f"{int(tick * 100)}%"
      elements.append(f"<text x='{px:.1f}' y='{y + height + 22}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>{label}</text>")
  elements.append(f"<text x='{x + width / 2}' y='{y + height + 43}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>{html.escape(x_label)}</text>")


def draw_axes(elements: list[str], x: float, y: float, width: float, height: float, y_min: float, y_max: float, *, percent_axis: bool) -> None:
  for index in range(5):
    value = y_min + (y_max - y_min) * index / 4
    py = y + height - index / 4 * height
    elements.append(f"<line x1='{x}' y1='{py:.1f}' x2='{x + width}' y2='{py:.1f}' stroke='{PALETTE['light_gray']}' stroke-width='1'/>")
    label = pct(value, digits=1) if percent_axis else f"{value:.2f}"
    elements.append(f"<text x='{x - 9}' y='{py + 4:.1f}' text-anchor='end' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>{label}</text>")
  elements.append(f"<rect x='{x}' y='{y}' width='{width}' height='{height}' fill='none' stroke='{PALETTE['gray']}' stroke-width='1'/>")


def bar_panel(
  elements: list[str],
  x: float,
  y: float,
  width: float,
  height: float,
  labels: list[str],
  values_: list[float],
  colors: list[str],
  title: str,
  format_type: str,
) -> None:
  maximum = max(values_) * 1.16 if values_ else 1.0
  elements.append(f"<text x='{x}' y='{y - 18}' font-family='Arial,sans-serif' font-size='17' font-weight='700' fill='{PALETTE['navy']}'>{html.escape(title)}</text>")
  for index in range(5):
    py = y + height - index / 4 * height
    elements.append(f"<line x1='{x}' y1='{py}' x2='{x + width}' y2='{py}' stroke='{PALETTE['light_gray']}'/>")
  slot = width / len(values_)
  for index, (label, value, color) in enumerate(zip(labels, values_, colors)):
    bar_width = slot * 0.58
    bar_height = value / max(maximum, 1e-12) * height
    px = x + index * slot + (slot - bar_width) / 2
    py = y + height - bar_height
    elements.append(f"<rect x='{px:.1f}' y='{py:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' fill='{color}' rx='2'/>")
    shown = pct(value, 2) if format_type == "percent" else (f"{value:,.0f}" if format_type == "integer" else f"{value:.1f}")
    elements.append(f"<text x='{px + bar_width / 2:.1f}' y='{py - 7:.1f}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' font-weight='700' fill='{PALETTE['ink']}'>{shown}</text>")
    elements.append(f"<text x='{px + bar_width / 2:.1f}' y='{y + height + 18}' transform='rotate(35 {px + bar_width / 2:.1f} {y + height + 18})' text-anchor='start' font-family='Arial,sans-serif' font-size='10' fill='{PALETTE['gray']}'>{html.escape(label)}</text>")
  elements.append(f"<rect x='{x}' y='{y}' width='{width}' height='{height}' fill='none' stroke='{PALETTE['gray']}'/>")


def grouped_bar_panel(
  elements: list[str],
  x: float,
  y: float,
  width: float,
  height: float,
  labels: list[str],
  series: list[tuple[str, list[float], str]],
  title: str,
  format_type: str,
) -> None:
  all_values = [value for _, values_, _ in series for value in values_]
  y_min, y_max = padded_range(all_values, include_zero=True)
  elements.append(f"<text x='{x}' y='{y - 18}' font-family='Arial,sans-serif' font-size='17' font-weight='700' fill='{PALETTE['navy']}'>{html.escape(title)}</text>")
  draw_axes(elements, x, y, width, height, y_min, y_max, percent_axis=format_type == "percent")
  slot = width / len(labels)
  group_width = slot * 0.7
  bar_width = group_width / len(series)
  zero_y = y + height - (0 - y_min) / max(1e-12, y_max - y_min) * height
  for series_index, (_, values_, color) in enumerate(series):
    for index, value in enumerate(values_):
      px = x + index * slot + (slot - group_width) / 2 + series_index * bar_width
      value_y = y + height - (value - y_min) / max(1e-12, y_max - y_min) * height
      top = min(zero_y, value_y)
      bar_height = max(1.0, abs(zero_y - value_y))
      elements.append(f"<rect x='{px:.1f}' y='{top:.1f}' width='{bar_width - 2:.1f}' height='{bar_height:.1f}' fill='{color}' opacity='.94'/>")
  for index, label in enumerate(labels):
    px = x + (index + 0.5) * slot
    elements.append(f"<text x='{px:.1f}' y='{y + height + 22}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>{html.escape(label)}</text>")


def confusion_panel(elements: list[str], x: float, y: float, size: float, matrix: list[list[int]], labels: list[str], title: str) -> None:
  cell = size / 3
  maximum = max(value for row in matrix for value in row)
  elements.append(f"<text x='{x}' y='{y - 20}' font-family='Arial,sans-serif' font-size='17' font-weight='700' fill='{PALETTE['navy']}'>{html.escape(title)}</text>")
  for row_index, row in enumerate(matrix):
    for column_index, value in enumerate(row):
      strength = value / max(1, maximum)
      shade = int(247 - strength * 150)
      fill = f"rgb({shade},{min(248, shade + 18)},{min(252, shade + 35)})"
      px = x + column_index * cell
      py = y + row_index * cell
      elements.append(f"<rect x='{px}' y='{py}' width='{cell}' height='{cell}' fill='{fill}' stroke='white' stroke-width='3'/>")
      elements.append(f"<text x='{px + cell / 2}' y='{py + cell / 2 + 7}' text-anchor='middle' font-family='Arial,sans-serif' font-size='22' font-weight='700' fill='{PALETTE['ink']}'>{value}</text>")
  for index, label in enumerate(labels):
    elements.append(f"<text x='{x + (index + .5) * cell}' y='{y + size + 25}' text-anchor='middle' font-family='Arial,sans-serif' font-size='12' fill='{PALETTE['gray']}'>{label}</text>")
    elements.append(f"<text x='{x - 18}' y='{y + (index + .5) * cell + 4}' text-anchor='middle' font-family='Arial,sans-serif' font-size='12' fill='{PALETTE['gray']}'>{label}</text>")
  elements.append(f"<text x='{x + size / 2}' y='{y + size + 48}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>predicted class</text>")
  elements.append(f"<text x='{x - 48}' y='{y + size / 2}' transform='rotate(-90 {x - 48} {y + size / 2})' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>actual class</text>")


def histogram_panel(
  elements: list[str],
  x: float,
  y: float,
  width: float,
  height: float,
  series: list[tuple[str, list[float], str]],
  title: str,
) -> None:
  combined = np.asarray([value for _, values_, _ in series for value in values_], dtype=float)
  lower, upper = np.quantile(combined, [0.005, 0.995])
  if lower == upper:
    lower -= 1e-6
    upper += 1e-6
  bins = np.linspace(lower, upper, 42)
  densities = []
  for _, values_, _ in series:
    clipped = np.clip(np.asarray(values_, dtype=float), lower, upper)
    counts, _ = np.histogram(clipped, bins=bins, density=True)
    densities.append(counts)
  maximum = max(float(np.max(counts)) for counts in densities)
  elements.append(f"<text x='{x}' y='{y - 18}' font-family='Arial,sans-serif' font-size='17' font-weight='700' fill='{PALETTE['navy']}'>{html.escape(title)}</text>")
  for (_, _, color), counts in zip(series, densities):
    points = []
    for index, count in enumerate(counts):
      px = x + (index + .5) / len(counts) * width
      py = y + height - float(count) / max(maximum, 1e-12) * height
      points.append(f"{px:.1f},{py:.1f}")
    elements.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2.2'/>")
  zero_x = x + (0 - lower) / (upper - lower) * width
  if x <= zero_x <= x + width:
    elements.append(f"<line x1='{zero_x:.1f}' y1='{y}' x2='{zero_x:.1f}' y2='{y + height}' stroke='{PALETTE['gray']}' stroke-dasharray='4 4'/>")
  elements.append(f"<rect x='{x}' y='{y}' width='{width}' height='{height}' fill='none' stroke='{PALETTE['gray']}'/>")
  for index, value in enumerate((lower, 0.0, upper)):
    px = x + (value - lower) / (upper - lower) * width
    if x <= px <= x + width:
      elements.append(f"<text x='{px:.1f}' y='{y + height + 21}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>{pct(value, 2)}</text>")
  elements.append(f"<text x='{x + width / 2}' y='{y + height + 43}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{PALETTE['gray']}'>five-minute net return</text>")


def add_legend(elements: list[str], x: float, y: float, entries: list[tuple[str, str]]) -> None:
  cursor = x
  for label, color in entries:
    elements.append(f"<line x1='{cursor}' y1='{y}' x2='{cursor + 30}' y2='{y}' stroke='{color}' stroke-width='4'/>")
    elements.append(f"<text x='{cursor + 38}' y='{y + 5}' font-family='Arial,sans-serif' font-size='12' fill='{PALETTE['ink']}'>{html.escape(label)}</text>")
    cursor += 48 + len(label) * 7.0


def padded_range(values_: list[float], *, include_zero: bool) -> tuple[float, float]:
  minimum = min(values_) if values_ else 0.0
  maximum = max(values_) if values_ else 1.0
  if include_zero:
    minimum = min(minimum, 0.0)
    maximum = max(maximum, 0.0)
  spread = maximum - minimum
  padding = spread * 0.1 if spread > 0 else max(abs(maximum) * 0.1, 0.1)
  return minimum - padding, maximum + padding


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def load_predictions(path: Path) -> list[dict[str, str]]:
  return read_csv(path)


def values(rows: list[dict[str, str]], field: str) -> list[float]:
  return [float(row[field]) for row in rows]


def parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def cumulative_equity(returns: Iterable[float]) -> list[float]:
  equity = 1.0
  output = []
  for value in returns:
    equity *= 1 + value
    output.append(equity)
  return output


def compounded_return(returns: Iterable[float]) -> float:
  equity = 1.0
  for value in returns:
    equity *= 1 + value
  return equity - 1.0


def drawdown_series(equity: list[float]) -> list[float]:
  peak = 1.0
  output = []
  for value in equity:
    peak = max(peak, value)
    output.append((peak - value) / peak if peak > 0 else 0.0)
  return output


def max_drawdown_duration(equity: list[float]) -> int:
  peak = 1.0
  duration = 0
  maximum = 0
  for value in equity:
    if value >= peak:
      peak = value
      duration = 0
    else:
      duration += 1
      maximum = max(maximum, duration)
  return maximum


def daily_compounded_returns(periods: list[dict[str, str]]) -> dict[str, float]:
  grouped: dict[str, list[float]] = defaultdict(list)
  for row in periods:
    grouped[row["candle_start"][:10]].append(float(row["net_return"]))
  return {date: compounded_return(returns) for date, returns in sorted(grouped.items())}


def lag_correlation(values_: list[float], lag: int) -> float:
  if len(values_) <= lag:
    return 0.0
  left = np.asarray(values_[:-lag], dtype=float)
  right = np.asarray(values_[lag:], dtype=float)
  if float(np.std(left)) == 0 or float(np.std(right)) == 0:
    return 0.0
  return float(np.corrcoef(left, right)[0, 1])


def pct(value: float, digits: int = 2) -> str:
  return f"{value * 100:.{digits}f}%"


if __name__ == "__main__":
  raise SystemExit(main())
