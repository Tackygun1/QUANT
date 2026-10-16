#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import generate_old_model_research_report as paper


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate the paper-style report for the calibrated DD1 hybrid direction model.")
  parser.add_argument("--config", type=Path, required=True)
  args = parser.parse_args()
  config = json.loads(args.config.read_text(encoding="utf-8"))

  output_dir = Path(config["output_dir"])
  figure_dir = output_dir / "figures"
  table_dir = output_dir / "tables"
  figure_dir.mkdir(parents=True, exist_ok=True)
  table_dir.mkdir(parents=True, exist_ok=True)

  common = {
    "bar_minutes": int(config["bar_minutes"]),
    "bootstrap_samples": int(config["bootstrap_samples"]),
    "bootstrap_block_bars": int(config["bootstrap_block_bars"]),
  }
  final_validation = paper.load_run(
    key="final_validation",
    label="Final model validation",
    split="final_model_validation",
    policy="original",
    root=Path(config["final_validation_backtest_dir"]),
    random_seed=int(config["random_seed"]),
    **common,
  )
  oof_original = paper.load_run(
    key="oof_original",
    label="Walk-forward OOF · Original",
    split="walk_forward_oof",
    policy="original",
    root=Path(config["oof_original_backtest_dir"]),
    random_seed=int(config["random_seed"]) + 1,
    **common,
  )
  oof_selected = paper.load_run(
    key="oof_selected",
    label="Walk-forward OOF · High-profit-per-trade",
    split="walk_forward_oof",
    policy="selected",
    root=Path(config["oof_selected_backtest_dir"]),
    random_seed=int(config["random_seed"]) + 2,
    **common,
  )
  forward_original = load_combined_run(
    key="forward_original",
    label="Forward policy holdout · Original",
    split="forward_policy_holdout",
    policy="original",
    roots=[Path(value) for value in config["forward_original_backtest_dirs"]],
    random_seed=int(config["random_seed"]) + 3,
    **common,
  )
  forward_selected = load_combined_run(
    key="forward_selected",
    label="Forward policy holdout · High-profit-per-trade",
    split="forward_policy_holdout",
    policy="selected",
    roots=[Path(value) for value in config["forward_selected_backtest_dirs"]],
    random_seed=int(config["random_seed"]) + 4,
    **common,
  )
  research_runs = [oof_original, oof_selected, forward_original, forward_selected]
  all_runs = [final_validation, *research_runs]
  for run in all_runs:
    add_active_event_tail_metrics(run)
  run_map = {run.key: run for run in all_runs}

  oof_predictions = paper.load_predictions(Path(config["oof_artifact_dir"]) / "validation_predictions.csv")
  final_predictions = paper.load_predictions(Path(config["final_artifact_dir"]) / "validation_predictions.csv")
  forward_predictions = []
  for artifact_dir in config["forward_prediction_artifact_dirs"]:
    forward_predictions.extend(paper.load_predictions(Path(artifact_dir) / "validation_predictions.csv"))
  classifications = {
    "walk_forward_oof": paper.classification_metrics(oof_predictions),
    "final_model_validation": paper.classification_metrics(final_predictions),
    "forward_policy_holdout": paper.classification_metrics(forward_predictions),
  }

  oof_summary = load_json(config["oof_evaluation_summary_path"])
  retrain_summary = load_json(config["retrain_summary_path"])
  tuning_summary = load_json(config["tuning_summary_path"])
  random_summary = load_json(config["random_baseline_summary_path"])
  forward_summary = load_json(config["forward_evaluation_summary_path"])
  simulation_rows = paper.read_csv(Path(config["random_baseline_simulations_path"]))
  fold_rows = build_fold_rows(
    paper.read_csv(Path(config["oof_artifact_dir"]) / "fold_metrics.csv"),
    tuning_summary,
  )

  fee_grid = [float(value) for value in config["fee_sensitivity_bps"]]
  fee_sensitivity = {
    run.key: paper.fixed_path_fee_sensitivity(run.periods, fee_grid)
    for run in research_runs
  }

  artifact_metadata = load_json(Path(config["final_artifact_dir"]) / "metadata.json")
  report_payload = {
    "title": config["title"],
    "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "config_path": str(args.config),
    "final_artifact_dir": config["final_artifact_dir"],
    "oof_artifact_dir": config["oof_artifact_dir"],
    "artifact_metadata": artifact_metadata,
    "methodology": {
      "bar_minutes": common["bar_minutes"],
      "bootstrap_samples": common["bootstrap_samples"],
      "bootstrap_block_bars": common["bootstrap_block_bars"],
      "bootstrap_method": "circular moving-block bootstrap of realized strategy bar returns",
      "fee_sensitivity_method": "fixed-position ex-post replay; signal path is held constant",
      "random_baseline_runs_per_mode": random_summary["simulations_per_baseline"],
      "forward_test_classification": forward_summary["test_classification"],
    },
    "runs": {run.key: run.metrics for run in all_runs},
    "classification": classifications,
    "folds": fold_rows,
    "fee_sensitivity": fee_sensitivity,
    "oof_evaluation": oof_summary,
    "retrain": retrain_summary,
    "tuning": tuning_summary,
    "random_baselines": random_summary,
    "forward_evaluation": forward_summary,
    "research_decision": "Retain the model as a research candidate; reject the selected fixed-25% execution policy for promotion pending a new virgin holdout.",
  }
  (output_dir / "research_metrics.json").write_text(
    json.dumps(report_payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )

  paper.write_performance_table(table_dir / "performance_metrics.csv", all_runs)
  paper.write_classification_table(table_dir / "classification_metrics.csv", classifications)
  paper.write_daily_table(table_dir / "daily_returns.csv", research_runs)
  paper.write_fee_table(table_dir / "fee_sensitivity.csv", fee_sensitivity)
  write_csv(table_dir / "walk_forward_folds.csv", fold_rows)
  write_random_table(table_dir / "random_baseline_summary.csv", random_summary)
  write_forward_window_table(table_dir / "forward_windows.csv", config)

  make_equity_figure(figure_dir / "figure_01_equity_curves.svg", run_map)
  make_drawdown_figure(figure_dir / "figure_02_drawdown.svg", run_map)
  make_execution_figure(figure_dir / "figure_03_execution_costs.svg", research_runs)
  make_fold_figure(figure_dir / "figure_04_walk_forward_folds.svg", fold_rows)
  make_classification_figure(figure_dir / "figure_05_prediction_diagnostics.svg", classifications)
  make_return_distribution_figure(figure_dir / "figure_06_return_distributions.svg", run_map)
  make_fee_figure(figure_dir / "figure_07_fee_sensitivity.svg", fee_sensitivity)
  make_exposure_figure(figure_dir / "figure_08_exposure.svg", run_map)
  make_random_figure(
    figure_dir / "figure_09_random_baselines.svg",
    simulation_rows,
    float(random_summary["reference_model"]["net_return"]),
  )

  markdown = build_report(
    config=config,
    runs=run_map,
    classifications=classifications,
    oof_summary=oof_summary,
    retrain_summary=retrain_summary,
    tuning_summary=tuning_summary,
    random_summary=random_summary,
    forward_summary=forward_summary,
    artifact_metadata=artifact_metadata,
  )
  (output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
  (output_dir / "REPORT.html").write_text(
    paper.build_html_report(markdown, config["title"]),
    encoding="utf-8",
  )
  print(json.dumps({
    "output_dir": str(output_dir),
    "report": str(output_dir / "REPORT.html"),
    "figures": 9,
    "tables": 7,
    "decision": report_payload["research_decision"],
  }, indent=2))
  return 0


def load_combined_run(
  *,
  key: str,
  label: str,
  split: str,
  policy: str,
  roots: list[Path],
  bar_minutes: int,
  bootstrap_samples: int,
  bootstrap_block_bars: int,
  random_seed: int,
) -> paper.BacktestRun:
  periods: list[dict[str, str]] = []
  trades: list[dict[str, str]] = []
  summaries: list[dict[str, Any]] = []
  for root in roots:
    periods.extend(paper.read_csv(root / "periods.csv"))
    trades.extend(paper.read_csv(root / "trades.csv"))
    summaries.append(load_json(root / "summary.json"))
  net = paper.values(periods, "net_return")
  active_net = [
    float(row["net_return"])
    for row in periods
    if abs(float(row["target_position"])) > 1e-12
  ]
  equity = paper.cumulative_equity(net)
  mean_return = float(np.mean(net)) if net else 0.0
  std_return = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
  periods_per_year = (60 / bar_minutes) * 24 * 365
  positive = sum(value for value in active_net if value > 0)
  negative = -sum(value for value in active_net if value < 0)
  aggregate = {
    "total_gross_return": sum(float(value["aggregate_metrics"]["total_gross_return"]) for value in summaries),
    "entry_count": sum(float(value["aggregate_metrics"]["entry_count"]) for value in summaries),
    "flip_count": sum(float(value["aggregate_metrics"].get("flip_count", 0.0)) for value in summaries),
  }
  summary = {
    "total_net_return": paper.compounded_return(net),
    "total_transaction_cost": sum(float(value["total_transaction_cost"]) for value in summaries),
    "total_turnover": sum(float(value["total_turnover"]) for value in summaries),
    "trades": sum(int(value["trades"]) for value in summaries),
    "active_periods": len(active_net),
    "active_hit_rate": sum(value > 0 for value in active_net) / len(active_net) if active_net else 0.0,
    "profit_factor_after_cost": positive / negative if negative > 0 else 0.0,
    "max_drawdown": max(paper.drawdown_series(equity), default=0.0),
    "sharpe_like": mean_return / std_return * math.sqrt(periods_per_year) if std_return > 0 else 0.0,
    "aggregate_metrics": aggregate,
  }
  metrics = paper.analyze_run(
    summary=summary,
    periods=periods,
    trades=trades,
    bar_minutes=bar_minutes,
    bootstrap_samples=bootstrap_samples,
    bootstrap_block_bars=bootstrap_block_bars,
    random_seed=random_seed,
  )
  return paper.BacktestRun(
    key=key,
    label=label,
    split=split,
    policy=policy,
    root=roots[0].parent,
    summary=summary,
    periods=periods,
    trades=trades,
    metrics=metrics,
  )


def build_fold_rows(metric_rows: list[dict[str, str]], tuning_summary: dict[str, Any]) -> list[dict[str, Any]]:
  original = tuning_summary["reference_metrics"]
  selected = tuning_summary["recommended_candidate"]
  rows = []
  for index, metric in enumerate(metric_rows):
    rows.append({
      "fold_index": int(metric["fold_index"]),
      "validation_start": metric["validation_start"],
      "validation_end": metric["validation_end"],
      "validation_rows": int(metric["validation_rows"]),
      "validation_accuracy": float(metric["validation_accuracy"]),
      "validation_macro_recall": float(metric["validation_macro_recall"]),
      "calibration_temperature": float(metric["calibration_temperature"]),
      "original_net_return": float(original["fold_returns"][index]),
      "selected_net_return": float(selected["fold_returns"][index]),
      "original_entries": int(original["fold_entries"][index]),
      "selected_entries": int(selected["fold_entries"][index]),
    })
  return rows


def add_active_event_tail_metrics(run: paper.BacktestRun) -> None:
  event_returns = [
    float(row["net_return"])
    for row in run.periods
    if abs(float(row["target_position"])) > 1e-12 or float(row["turnover"]) > 1e-12
  ]
  if not event_returns:
    run.metrics["active_event_var_95"] = 0.0
    run.metrics["active_event_cvar_95"] = 0.0
    run.metrics["active_event_observations"] = 0
    return
  threshold = float(np.quantile(event_returns, 0.05))
  tail = [value for value in event_returns if value <= threshold]
  run.metrics["active_event_var_95"] = -threshold
  run.metrics["active_event_cvar_95"] = -float(np.mean(tail))
  run.metrics["active_event_observations"] = len(event_returns)


def make_equity_figure(path: Path, runs: dict[str, paper.BacktestRun]) -> None:
  elements = paper.svg_canvas("Figure 1. Strategy and benchmark equity", "Compounded growth of $1; execution state resets only at the forward-window boundary")
  for panel, (title, original_key, selected_key) in enumerate([
    ("Walk-forward OOF", "oof_original", "oof_selected"),
    ("Forward execution-policy test", "forward_original", "forward_selected"),
  ]):
    original, selected = runs[original_key], runs[selected_key]
    series = [
      ("Original", paper.cumulative_equity(paper.values(original.periods, "net_return")), paper.PALETTE["blue"]),
      ("Selected", paper.cumulative_equity(paper.values(selected.periods, "net_return")), paper.PALETTE["teal"]),
      ("BTC buy-and-hold", paper.cumulative_equity(paper.values(original.periods, "realized_return")), paper.PALETTE["gray"]),
    ]
    paper.line_panel(elements, 70 + panel * 580, 120, 520, 430, series, title, percent_axis=False)
  paper.add_legend(elements, 390, 610, [("Original", paper.PALETTE["blue"]), ("Selected", paper.PALETTE["teal"]), ("BTC", paper.PALETTE["gray"])])
  paper.save_svg(path, elements)


def make_drawdown_figure(path: Path, runs: dict[str, paper.BacktestRun]) -> None:
  elements = paper.svg_canvas("Figure 2. Drawdown paths", "Peak-to-trough portfolio loss; lower is worse and the configured halt threshold is 1%")
  for panel, (title, original_key, selected_key) in enumerate([
    ("Walk-forward OOF", "oof_original", "oof_selected"),
    ("Forward execution-policy test", "forward_original", "forward_selected"),
  ]):
    series = []
    for label, key, color in [("Original", original_key, paper.PALETTE["blue"]), ("Selected", selected_key, paper.PALETTE["teal"])]:
      equity = paper.cumulative_equity(paper.values(runs[key].periods, "net_return"))
      series.append((label, [-value for value in paper.drawdown_series(equity)], color))
    paper.line_panel(elements, 70 + panel * 580, 120, 520, 430, series, title, percent_axis=True, include_zero=True)
  paper.add_legend(elements, 430, 610, [("Original", paper.PALETTE["blue"]), ("Selected", paper.PALETTE["teal"])])
  paper.save_svg(path, elements)


def make_execution_figure(path: Path, runs: list[paper.BacktestRun]) -> None:
  elements = paper.svg_canvas("Figure 3. Execution intensity and implementation costs", "Costs assume 1.0 bp fee plus 0.5 bp slippage per unit of turnover")
  labels = ["OOF orig", "OOF selected", "Fwd orig", "Fwd selected"]
  colors = [paper.PALETTE["blue"], paper.PALETTE["teal"], paper.PALETTE["blue"], paper.PALETTE["teal"]]
  paper.bar_panel(elements, 55, 135, 350, 400, labels, [run.metrics["entries"] for run in runs], colors, "Position entries", "integer")
  paper.bar_panel(elements, 425, 135, 350, 400, labels, [run.metrics["turnover"] for run in runs], colors, "Total turnover", "decimal")
  paper.bar_panel(elements, 795, 135, 350, 400, labels, [run.metrics["transaction_cost"] for run in runs], colors, "Modeled cost", "percent")
  paper.add_legend(elements, 430, 610, [("Original", paper.PALETTE["blue"]), ("Selected", paper.PALETTE["teal"])])
  paper.save_svg(path, elements)


def make_fold_figure(path: Path, folds: list[dict[str, Any]]) -> None:
  elements = paper.svg_canvas("Figure 4. Walk-forward fold stability", "Non-overlapping 1,024-bar validation folds; execution parameters were selected on this OOF stream")
  paper.grouped_bar_panel(
    elements,
    90,
    130,
    1020,
    430,
    [f"F{row['fold_index']}" for row in folds],
    [
      ("Original", [row["original_net_return"] for row in folds], paper.PALETTE["blue"]),
      ("Selected", [row["selected_net_return"] for row in folds], paper.PALETTE["teal"]),
    ],
    "Net return by fold",
    "percent",
  )
  paper.add_legend(elements, 430, 620, [("Original", paper.PALETTE["blue"]), ("Selected", paper.PALETTE["teal"])])
  paper.save_svg(path, elements)


def make_classification_figure(path: Path, classifications: dict[str, dict[str, Any]]) -> None:
  elements = paper.svg_canvas("Figure 5. Prediction stability across evaluation layers", "Class labels: −1 down, 0 neutral, +1 up")
  classes = ["−1", "0", "+1"]
  series = []
  for label, key, color in [
    ("OOF", "walk_forward_oof", paper.PALETTE["blue"]),
    ("Final validation", "final_model_validation", paper.PALETTE["teal"]),
    ("Forward", "forward_policy_holdout", paper.PALETTE["red"]),
  ]:
    metrics = classifications[key]
    series.append((label, [metrics["prediction_distribution"][str(cls)] / metrics["rows"] for cls in (-1, 0, 1)], color))
  paper.grouped_bar_panel(elements, 55, 140, 550, 390, classes, series, "Prediction class share", "percent")
  forward = classifications["forward_policy_holdout"]
  paper.confusion_panel(elements, 690, 150, 380, forward["confusion_matrix"], classes, "Forward confusion matrix")
  paper.add_legend(elements, 300, 620, [("OOF", paper.PALETTE["blue"]), ("Final validation", paper.PALETTE["teal"]), ("Forward", paper.PALETTE["red"])])
  paper.save_svg(path, elements)


def make_return_distribution_figure(path: Path, runs: dict[str, paper.BacktestRun]) -> None:
  elements = paper.svg_canvas("Figure 6. Distribution of realized strategy returns", "Five-minute net returns; display range is clipped to joint 0.5–99.5% quantiles")
  for panel, (title, original_key, selected_key) in enumerate([
    ("Walk-forward OOF", "oof_original", "oof_selected"),
    ("Forward execution-policy test", "forward_original", "forward_selected"),
  ]):
    paper.histogram_panel(
      elements,
      65 + panel * 575,
      130,
      500,
      410,
      [
        ("Original", paper.values(runs[original_key].periods, "net_return"), paper.PALETTE["blue"]),
        ("Selected", paper.values(runs[selected_key].periods, "net_return"), paper.PALETTE["teal"]),
      ],
      title,
    )
  paper.add_legend(elements, 430, 620, [("Original", paper.PALETTE["blue"]), ("Selected", paper.PALETTE["teal"])])
  paper.save_svg(path, elements)


def make_fee_figure(path: Path, sensitivity: dict[str, list[dict[str, float]]]) -> None:
  elements = paper.svg_canvas("Figure 7. Fixed-path transaction-cost sensitivity", "Combined fee plus slippage; positions are held fixed")
  for panel, (title, original_key, selected_key) in enumerate([
    ("Walk-forward OOF", "oof_original", "oof_selected"),
    ("Forward execution-policy test", "forward_original", "forward_selected"),
  ]):
    x_values = [row["total_cost_bps"] for row in sensitivity[original_key]]
    series = [
      ("Original", [row["net_return"] for row in sensitivity[original_key]], paper.PALETTE["blue"]),
      ("Selected", [row["net_return"] for row in sensitivity[selected_key]], paper.PALETTE["teal"]),
    ]
    paper.line_panel(elements, 70 + panel * 580, 120, 520, 430, series, title, percent_axis=True, include_zero=True, x_values=x_values, x_label="total cost (bps)")
  paper.add_legend(elements, 430, 610, [("Original", paper.PALETTE["blue"]), ("Selected", paper.PALETTE["teal"])])
  paper.save_svg(path, elements)


def make_exposure_figure(path: Path, runs: dict[str, paper.BacktestRun]) -> None:
  elements = paper.svg_canvas("Figure 8. Position exposure through time", "Target position after execution and risk constraints; positive is long")
  for panel, (title, original_key, selected_key) in enumerate([
    ("Walk-forward OOF", "oof_original", "oof_selected"),
    ("Forward execution-policy test", "forward_original", "forward_selected"),
  ]):
    series = [
      ("Original", paper.values(runs[original_key].periods, "target_position"), paper.PALETTE["blue"]),
      ("Selected", paper.values(runs[selected_key].periods, "target_position"), paper.PALETTE["teal"]),
    ]
    paper.line_panel(elements, 70 + panel * 580, 120, 520, 430, series, title, percent_axis=False, include_zero=True, y_fixed=(-1.05, 1.05))
  paper.add_legend(elements, 430, 610, [("Original", paper.PALETTE["blue"]), ("Selected", paper.PALETTE["teal"])])
  paper.save_svg(path, elements)


def make_random_figure(path: Path, simulations: list[dict[str, str]], reference: float) -> None:
  elements = paper.svg_canvas("Figure 9. Random and shuffled-signal benchmark", "Distribution of 500 OOF net returns per baseline; red line is the model's original-policy OOF return")
  specs = [
    ("Uniform random class", "uniform_random_class"),
    ("Shuffled model signal", "shuffled_model_signal"),
  ]
  for panel, (title, mode) in enumerate(specs):
    values = [float(row["net_return"]) for row in simulations if row["mode"] == mode]
    random_histogram_panel(elements, 70 + panel * 580, 130, 520, 410, values, reference, title)
  paper.add_legend(elements, 430, 610, [("Random distribution", paper.PALETTE["gray"]), ("Model OOF", paper.PALETTE["red"])])
  paper.save_svg(path, elements)


def random_histogram_panel(
  elements: list[str],
  x: float,
  y: float,
  width: float,
  height: float,
  values: list[float],
  reference: float,
  title: str,
) -> None:
  data = np.asarray(values, dtype=float)
  low = min(float(np.quantile(data, 0.005)), reference)
  high = max(float(np.quantile(data, 0.995)), reference)
  span = max(high - low, 1e-6)
  low -= span * 0.05
  high += span * 0.05
  counts, edges = np.histogram(data, bins=np.linspace(low, high, 42), density=True)
  maximum = max(float(np.max(counts)), 1e-12)
  elements.append(f"<text x='{x}' y='{y - 18}' font-family='Arial,sans-serif' font-size='17' font-weight='700' fill='{paper.PALETTE['navy']}'>{html.escape(title)}</text>")
  points = []
  for index, count in enumerate(counts):
    px = x + ((edges[index] + edges[index + 1]) / 2 - low) / (high - low) * width
    py = y + height - float(count) / maximum * height
    points.append(f"{px:.1f},{py:.1f}")
  elements.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{paper.PALETTE['gray']}' stroke-width='2.3'/>")
  reference_x = x + (reference - low) / (high - low) * width
  elements.append(f"<line x1='{reference_x:.1f}' y1='{y}' x2='{reference_x:.1f}' y2='{y + height}' stroke='{paper.PALETTE['red']}' stroke-width='3' stroke-dasharray='7 5'/>")
  elements.append(f"<rect x='{x}' y='{y}' width='{width}' height='{height}' fill='none' stroke='{paper.PALETTE['gray']}'/>")
  for tick in (low, (low + high) / 2, high):
    px = x + (tick - low) / (high - low) * width
    elements.append(f"<text x='{px:.1f}' y='{y + height + 22}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{paper.PALETTE['gray']}'>{paper.pct(tick, 1)}</text>")
  elements.append(f"<text x='{x + width / 2}' y='{y + height + 45}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{paper.PALETTE['gray']}'>net return</text>")


def build_report(
  *,
  config: dict[str, Any],
  runs: dict[str, paper.BacktestRun],
  classifications: dict[str, dict[str, Any]],
  oof_summary: dict[str, Any],
  retrain_summary: dict[str, Any],
  tuning_summary: dict[str, Any],
  random_summary: dict[str, Any],
  forward_summary: dict[str, Any],
  artifact_metadata: dict[str, Any],
) -> str:
  original = runs["oof_original"].metrics
  selected = runs["oof_selected"].metrics
  forward_original = runs["forward_original"].metrics
  forward_selected = runs["forward_selected"].metrics
  final_validation = runs["final_validation"].metrics
  oof_class = classifications["walk_forward_oof"]
  final_class = classifications["final_model_validation"]
  forward_class = classifications["forward_policy_holdout"]
  reference = tuning_summary["reference_metrics"]
  candidate = tuning_summary["recommended_candidate"]
  random_uniform = random_summary["baselines"]["uniform_random_class"]
  random_shuffled = random_summary["baselines"]["shuffled_model_signal"]
  adjustment_reduction = 1 - selected["position_adjustments"] / original["position_adjustments"]
  entry_reduction = 1 - selected["entries"] / original["entries"]
  forward_adjustment_reduction = 1 - forward_selected["position_adjustments"] / forward_original["position_adjustments"]
  forward_cost_reduction = 1 - forward_selected["transaction_cost"] / forward_original["transaction_cost"]
  rows = [
    ("Walk-forward OOF / original", original),
    ("Walk-forward OOF / selected", selected),
    ("Forward policy holdout / original", forward_original),
    ("Forward policy holdout / selected", forward_selected),
  ]
  table = [
    "| Evaluation and execution | Gross return | Net return | Costs | Entries | Adjustments | Max DD | Sharpe-like |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for label, metrics in rows:
    table.append(
      f"| {label} | {paper.pct(metrics['gross_return'])} | {paper.pct(metrics['net_return'])} | "
      f"{paper.pct(metrics['transaction_cost'])} | {metrics['entries']:,} | {metrics['position_adjustments']:,} | "
      f"{paper.pct(metrics['max_drawdown'])} | {metrics['sharpe_like_annualized']:.2f} |"
    )
  calibration = artifact_metadata["extra"]["probability_calibration"]
  report = f"""# {config['title']}

## Abstract

This report audits the newly retrained `lightgbm_hybrid_direction_meta` classifier and its cost-aware execution policies. Leakage-audited walk-forward OOF evaluation produced {paper.pct(original['net_return'])} after costs with {paper.pct(original['max_drawdown'])} maximum drawdown, but {paper.pct(oof_summary['robustness']['best_fold_share_of_total_net_return'])} of total OOF profit came from the best fold. A post-hoc high-profit-per-trade policy raised selected-sample net return to {paper.pct(selected['net_return'])}, increased net return per entry from {reference['net_profit_per_entry_bps']:.2f} to {candidate['net_profit_per_entry_bps']:.2f} bps, and reduced entries by {paper.pct(entry_reduction)}. That result did not survive the later execution-policy replay: the selected policy returned {paper.pct(forward_selected['net_return'])} versus {paper.pct(forward_original['net_return'])} for the original policy. The model remains a credible research candidate, but the fixed-25% policy is rejected for deployment and a genuinely untouched prospective holdout is still required.

## 1. Experimental design and evidence hierarchy

- **Instrument and frequency:** Coinbase BTC-USD, five-minute bars.
- **Model:** `{artifact_metadata['model_name']}` with `{len(artifact_metadata['feature_columns'])}` market, regime, structure, trade-flow, and stacked transformer features.
- **Target:** `{artifact_metadata['target_column']}` with short, neutral, and long classes.
- **Walk-forward OOF:** {original['bars']:,} observations from {original['start_time']} through {original['end_time']} ({original['calendar_days']:.1f} calendar days), using {oof_summary['protocol']['folds']} non-overlapping folds with one-row purge and embargo.
- **Final-model validation:** {final_validation['bars']:,} observations from {final_validation['start_time']} through {final_validation['end_time']}; this is model-development evidence, not a virgin test.
- **Forward policy replay:** {forward_original['bars']:,} observations from {forward_original['start_time']} through {forward_original['end_time']} ({forward_original['calendar_days']:.1f} elapsed days).
- **Implementation costs:** 1.0 bp fee plus 0.5 bp slippage per unit of turnover; the selected policy was also re-simulated at 3.0 bps total cost.
- **Risk control:** 1% daily-loss and portfolio-drawdown halt thresholds. A halt is not a guarantee against discrete bar or execution-gap overshoot.
- **Important qualification:** the forward datasets were not used to select the new execution parameters, but had been inspected during earlier model diagnosis. They are parameter-out-of-sample, not a final virgin model holdout.
- **Strict holdout status:** a frozen August 11-18 configuration exists, but retrieval was stopped before scoring because exact trade-flow history required impractical sequential pagination. Missing trade features were not approximated.

## 2. Headline economic results

{chr(10).join(table)}

The selected OOF policy uses a 0.15 probability-margin threshold, a nine-bar minimum hold, a 0.10 rebalance deadband, and fixed 25% positions. OOF adjustments fell {paper.pct(adjustment_reduction)}, from {original['position_adjustments']:,} to {selected['position_adjustments']:,}, but turnover changed from {original['turnover']:.2f} to {selected['turnover']:.2f}. Fewer decisions therefore did not mean lower capital turnover: accepted signals were much larger.

![Equity curves](figures/figure_01_equity_curves.svg)

![Drawdown paths](figures/figure_02_drawdown.svg)

## 3. Walk-forward stability and selection risk

The original policy was profitable in {reference['positive_folds']} of {len(reference['fold_returns'])} folds. Its median fold return was {paper.pct(reference['median_fold_return'], 3)}, and the best fold supplied {paper.pct(reference['best_fold_profit_share'])} of summed positive fold profit. Excluding the best fold reduced aggregate OOF return to approximately {paper.pct(oof_summary['robustness']['net_return_excluding_best_fold'])}.

The selected policy improved the count to {candidate['positive_folds']} profitable folds, raised median fold return to {paper.pct(candidate['median_fold_return'], 3)}, and reduced best-fold profit concentration to {paper.pct(candidate['best_fold_profit_share'])}. However, these execution parameters were chosen on the same OOF stream used to report those improvements, so the result is selection-biased. Its 95% circular block-bootstrap interval is {paper.pct(selected['bootstrap_total_return_ci_95'][0])} to {paper.pct(selected['bootstrap_total_return_ci_95'][1])}; this conditional interval does not account for searching multiple execution variants.

![Walk-forward folds](figures/figure_04_walk_forward_folds.svg)

## 4. Predictive diagnostics and calibration

OOF accuracy was {paper.pct(oof_class['accuracy'])}, balanced accuracy was {paper.pct(oof_class['balanced_accuracy'])}, and Matthews correlation was {oof_class['matthews_correlation']:.3f}. Majority-class accuracy was {paper.pct(oof_summary['predictive_metrics']['majority_class_accuracy'])}, giving an accuracy lift of {paper.pct(oof_summary['predictive_metrics']['accuracy_lift_over_majority'])}.

The final fit used {retrain_summary['training']['fit_rows']:,} fit rows plus {retrain_summary['training']['calibration_rows']:,} calibration rows. Temperature scaling selected a temperature of {calibration['temperature']:.3f}. Final validation accuracy was {paper.pct(final_class['accuracy'])}, balanced accuracy was {paper.pct(final_class['balanced_accuracy'])}, and Matthews correlation was {final_class['matthews_correlation']:.3f}. On the later forward data, accuracy was {paper.pct(forward_class['accuracy'])}, balanced accuracy was {paper.pct(forward_class['balanced_accuracy'])}, and Matthews correlation was {forward_class['matthews_correlation']:.3f}.

All {forward_class['rows']:,} forward hard-class predictions were neutral. This does not imply a no-trade policy: execution uses the signed probability margin rather than the argmax hard class, and the margin still generated eighteen original-policy entries and five selected-policy entries. The hard-class collapse nevertheless signals weak class separation and explains the chance-level balanced accuracy.

![Prediction diagnostics](figures/figure_05_prediction_diagnostics.svg)

## 5. Random and shuffled-signal controls

Two 500-run null experiments were evaluated on exactly the same OOF timestamps and cost model. Uniform random classes had median net return {paper.pct(random_uniform['distributions']['net_return']['median'])}; only {paper.pct(random_uniform['comparison_to_reference']['profitable_run_fraction'])} of runs were profitable. Shuffled model signals had median net return {paper.pct(random_shuffled['distributions']['net_return']['median'])}; only {paper.pct(random_shuffled['comparison_to_reference']['profitable_run_fraction'])} were profitable.

The one-sided empirical p-values for attaining at least the model's original OOF return were {random_uniform['comparison_to_reference']['one_sided_empirical_p_net_return']:.4f} for uniform random classes and {random_shuffled['comparison_to_reference']['one_sided_empirical_p_net_return']:.4f} for shuffled signals. This supports the presence of historical timing information in the OOF predictions, but it does not establish stable future profitability or validate the subsequently selected execution overlay.

![Random baselines](figures/figure_09_random_baselines.svg)

## 6. Execution efficiency and fee sensitivity

The selected policy increased OOF net profit per entry from {reference['net_profit_per_entry_bps']:.2f} to {candidate['net_profit_per_entry_bps']:.2f} bps and net edge per unit turnover from {reference['net_edge_per_turnover_bps']:.2f} to {candidate['net_edge_per_turnover_bps']:.2f} bps. At 3.0 bps total transaction cost, it retained {paper.pct(candidate['stress_net_return'])} net return with {paper.pct(candidate['stress_max_drawdown'])} maximum drawdown.

In the forward replay, the policy still traded: five entries versus eighteen originally. Adjustments fell {paper.pct(forward_adjustment_reduction)}, yet modeled cost fell only {paper.pct(forward_cost_reduction)} because each selected round trip moved 25% of the portfolio. One adverse short position contributed -20.74 portfolio bps and dominated the forward result. The selected policy's net return per entry was {forward_summary['selected_1p5bps']['net_return_per_entry_bps']:.2f} bps, versus {forward_summary['original_1p5bps']['net_return_per_entry_bps']:.2f} bps originally.

![Execution costs](figures/figure_03_execution_costs.svg)

![Fee sensitivity](figures/figure_07_fee_sensitivity.svg)

## 7. Risk, tails, and exposure

All reported OOF and forward maximum drawdowns remained below the configured 1% ceiling. The selected OOF maximum drawdown was {paper.pct(selected['max_drawdown'])}; under 3.0 bps cost it was {paper.pct(candidate['stress_max_drawdown'])}. The combined forward selected drawdown was {paper.pct(forward_selected['max_drawdown'])}.

Across the {selected['active_event_observations']:,} OOF bars with exposure or turnover, selected-policy 95% historical VaR and expected shortfall were {paper.pct(selected['active_event_var_95'], 3)} and {paper.pct(selected['active_event_cvar_95'], 3)}. Its flat exposure share was {paper.pct(selected['flat_share'])}, versus {paper.pct(original['flat_share'])} under the original execution. Sparse exposure controls market participation, but fixed sizing creates concentration conditional on entry.

![Return distributions](figures/figure_06_return_distributions.svg)

![Position exposure](figures/figure_08_exposure.svg)

## 8. Statistical and methodological limitations

1. The high-profit-per-trade policy was chosen on the OOF stream it is scored on. Its OOF return is exploratory and optimistically biased.
2. The original OOF profit is highly fold-concentrated; a positive aggregate alone is not strong evidence of temporal stability.
3. The forward policy replay contains only five selected entries and uses datasets previously inspected during model diagnosis. It can reject the selected policy, but cannot serve as final untouched confirmation of the model.
4. The 1% drawdown setting is an execution halt threshold, not a hard mathematical loss bound.
5. The proportional cost model omits spread variation, market impact, latency, rejected orders, fee tiers, funding, and taxes.
6. Fixed-path fee sensitivity holds positions constant. It is an implementation-shortfall analysis, not a full behavioral re-simulation.
7. Annualized Sharpe-like and Sortino-like values are unstable over these short and sparse samples.
8. Random-baseline p-values are empirical comparisons to two specified null processes; they are not universal proof against data mining.

## 9. Conclusion

The retrained classifier is materially better documented than the legacy artifact: it has purged walk-forward predictions, explicit probability calibration, a 1% risk halt, fee stress, and random-signal controls. The OOF evidence is positive but inconclusive because the original profit is concentrated and the improved execution overlay was selected post hoc. The later replay shows that fixed 25% positions overcorrected the fee problem: trade events fell sharply, but turnover barely changed and profitability reversed.

The defensible project conclusion is **retain the calibrated model as a research candidate, reject the fixed-25% selected policy for promotion, and require a newly frozen prospective holdout before any deployment claim**. A future OOF-only search should preserve the 0.15 entry margin and 0.10 rebalance deadband as hypotheses while returning to smaller volatility-scaled sizing.

## Reproducibility appendix

- Generator configuration: `configs/research/runtime/new_hybrid_direction_research_report_20260818.json`.
- Generator: `scripts/research/generate_new_model_research_report.py`.
- Machine-readable metrics: `research_metrics.json`.
- Tabular exports: `tables/performance_metrics.csv`, `tables/classification_metrics.csv`, `tables/daily_returns.csv`, `tables/walk_forward_folds.csv`, `tables/fee_sensitivity.csv`, `tables/random_baseline_summary.csv`, and `tables/forward_windows.csv`.
- Figures are standalone SVG vector graphics suitable for browser display and typesetting systems.
"""
  return report


def write_random_table(path: Path, summary: dict[str, Any]) -> None:
  rows = []
  for mode, payload in summary["baselines"].items():
    net = payload["distributions"]["net_return"]
    comparison = payload["comparison_to_reference"]
    rows.append({
      "mode": mode,
      "runs": payload["runs"],
      "mean_net_return": net["mean"],
      "median_net_return": net["median"],
      "p025_net_return": net["p025"],
      "p975_net_return": net["p975"],
      "profitable_run_fraction": comparison["profitable_run_fraction"],
      "one_sided_empirical_p_net_return": comparison["one_sided_empirical_p_net_return"],
      "drawdown_limit_breach_fraction": comparison["drawdown_limit_breach_fraction"],
    })
  write_csv(path, rows)


def write_forward_window_table(path: Path, config: dict[str, Any]) -> None:
  rows = []
  for policy in ("original", "selected"):
    for index, root_value in enumerate(config[f"forward_{policy}_backtest_dirs"], start=1):
      root = Path(root_value)
      summary = load_json(root / "summary.json")
      aggregate = summary["aggregate_metrics"]
      periods = paper.read_csv(root / "periods.csv")
      rows.append({
        "policy": policy,
        "window": index,
        "start_time": periods[0]["candle_start"],
        "end_time": periods[-1]["candle_start"],
        "rows": len(periods),
        "gross_return": aggregate["total_gross_return"],
        "net_return": aggregate["total_net_return"],
        "transaction_cost": aggregate["total_transaction_cost"],
        "turnover": aggregate["total_turnover"],
        "entries": int(aggregate["entry_count"]),
        "position_adjustments": int(aggregate["trade_count"]),
        "max_drawdown": aggregate["max_drawdown"],
      })
  write_csv(path, rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    path.write_text("", encoding="utf-8")
    return
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def load_json(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
  raise SystemExit(main())
