#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

# The reused SVG helpers do not require SciPy, while the older report module imports
# it for unrelated skew/kurtosis calculations. Keep this supplement runnable in the
# repository's minimal environment without altering those helpers.
try:
  import scipy  # noqa: F401
except ModuleNotFoundError:
  scipy_stub = types.ModuleType("scipy")
  scipy_stub.stats = types.SimpleNamespace()
  sys.modules["scipy"] = scipy_stub

import generate_old_model_research_report as paper


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/research/new_hybrid_direction_model_20260818"
TABLES = REPORT / "tables"
FIGURES = REPORT / "figures"
RETRAIN = ROOT / "experiments/retrains/coinbase/BTC-USD/20260817T133422Z_calibrated_dd1"
FEATURE_ROOT = (
  ROOT
  / "experiments/retrains/coinbase/BTC-USD/20260607T195625Z_supply_zones/features/training_datasets"
  / "market_hybrid_stacked_meta_only_zones/coinbase/BTC-USD"
)
FINAL_ARTIFACT = RETRAIN / "models/ensembles/lightgbm_hybrid_direction_meta/coinbase/BTC-USD/20260817T133444Z"
OOF_ARTIFACT = RETRAIN / "models/walk_forward/meta_models/lightgbm_hybrid_direction_meta/coinbase/BTC-USD/20260818T004349Z"
STAGE1_SUMMARY = (
  ROOT
  / "experiments/retrains/coinbase/BTC-USD/20260607T195625Z_supply_zones/ablations/no_zone/models/walk_forward"
  / "hybrid_transformers/quantile_forecaster/coinbase/BTC-USD/20260607T214628Z/summary.json"
)
TUNING_SUMMARY = RETRAIN / "reports/profit_per_trade_tuning_boundary/20260818T012122Z/summary.json"
OOF_PERIODS = RETRAIN / "reports/oof_final/lightgbm_hybrid_direction_meta/coinbase/BTC-USD/20260818T004408Z/periods.csv"
FINAL_PERIODS = RETRAIN / "reports/validation/lightgbm_hybrid_direction_meta/coinbase/BTC-USD/20260817T133527Z/periods.csv"
FORWARD_PREDICTIONS = [
  ROOT / "experiments/out_of_sample/coinbase/BTC-USD/frozen_margin_fast_strict_promoted_20260422T005857Z/evaluations/lightgbm_hybrid_direction_meta/20260817T133444Z/validation_predictions.csv",
  ROOT / "experiments/out_of_sample/coinbase/BTC-USD/fresh_100k_20260422T020345Z/evaluations/lightgbm_hybrid_direction_meta/20260817T133444Z/validation_predictions.csv",
]
FORWARD_PERIODS = [
  RETRAIN / "reports/untouched_execution_test_20260818/20260416_to_20260420/original_1p5/lightgbm_hybrid_direction_meta/coinbase/BTC-USD/20260818T024743Z/periods.csv",
  RETRAIN / "reports/untouched_execution_test_20260818/20260420_to_20260422/original_1p5/lightgbm_hybrid_direction_meta/coinbase/BTC-USD/20260818T024743Z/periods.csv",
]

COLORS = {
  "blue": "#2878b5",
  "teal": "#2a9d8f",
  "amber": "#e9a23b",
  "red": "#d1495b",
  "gray": "#67727e",
  "navy": "#17324d",
  "ink": "#18212b",
  "light": "#d9e0e6",
  "paper": "#fbfaf7",
  "green": "#3a8f5b",
}


def main() -> int:
  TABLES.mkdir(parents=True, exist_ok=True)
  FIGURES.mkdir(parents=True, exist_ok=True)
  metadata = load_json(FINAL_ARTIFACT / "metadata.json")
  research = load_json(REPORT / "research_metrics.json")
  stage1_rows = build_stage1_rows(load_json(STAGE1_SUMMARY))
  write_csv(TABLES / "technical_stage1_horizons.csv", stage1_rows)

  forward_period_rows = [row for path in FORWARD_PERIODS for row in read_csv(path)]
  forward_margin_by_time = {row["candle_start"]: row["prediction"] for row in forward_period_rows}
  forward_prediction_rows = [row for path in FORWARD_PREDICTIONS for row in read_csv(path)]
  for row in forward_prediction_rows:
    row["prediction_margin"] = forward_margin_by_time.get(row["candle_start"], "0")
  datasets = {
    "walk_forward_oof": read_csv(OOF_ARTIFACT / "validation_predictions.csv"),
    "final_model_validation": read_csv(FINAL_ARTIFACT / "validation_predictions.csv"),
    "forward_policy_replay": forward_prediction_rows,
  }
  probability_rows, calibration_rows, confusion_rows, threshold_rows = build_probability_diagnostics(datasets)
  write_csv(TABLES / "technical_probability_diagnostics.csv", probability_rows)
  write_csv(TABLES / "technical_calibration_bins.csv", calibration_rows)
  write_csv(TABLES / "technical_confusion_matrices.csv", confusion_rows)
  write_csv(TABLES / "technical_margin_thresholds.csv", threshold_rows)

  completeness = feature_completeness(metadata["feature_columns"])
  importance = feature_importance(metadata["feature_columns"], FINAL_ARTIFACT / "model.pkl", completeness)
  group_importance = aggregate_group_importance(importance)
  write_csv(TABLES / "technical_feature_importance.csv", importance)
  write_csv(TABLES / "technical_feature_group_importance.csv", group_importance)

  folds = read_csv(REPORT / "tables/walk_forward_folds.csv")
  correlation_rows, concentration_rows = fold_diagnostics(folds)
  write_csv(TABLES / "technical_fold_correlations.csv", correlation_rows)
  write_csv(TABLES / "technical_fold_concentration.csv", concentration_rows)

  tuning = load_json(TUNING_SUMMARY)
  policy_rows = policy_grid_rows(tuning)
  write_csv(TABLES / "technical_policy_grid.csv", policy_rows)

  daily_rows = read_csv(REPORT / "tables/daily_returns.csv")
  daily_summary = daily_diagnostics(daily_rows)
  write_csv(TABLES / "technical_daily_summary.csv", daily_summary)

  period_sets = {
    "walk_forward_oof": read_csv(OOF_PERIODS),
    "final_model_validation": read_csv(FINAL_PERIODS),
    "forward_policy_replay": forward_period_rows,
  }
  margin_deciles = build_margin_deciles(datasets, period_sets)
  write_csv(TABLES / "technical_margin_deciles.csv", margin_deciles)

  split_rows = build_split_rows(research)
  write_csv(TABLES / "technical_evidence_partitions.csv", split_rows)

  make_stage1_figure(FIGURES / "figure_10_upstream_horizon_metrics.svg", stage1_rows)
  make_reliability_figure(FIGURES / "figure_11_probability_reliability.svg", calibration_rows)
  make_importance_figure(FIGURES / "figure_12_feature_importance.svg", importance[:20])
  make_group_importance_figure(FIGURES / "figure_13_feature_group_importance.svg", group_importance)
  make_margin_figure(FIGURES / "figure_14_margin_thresholds.svg", threshold_rows)
  make_policy_grid_figure(FIGURES / "figure_15_policy_search_surface.svg", policy_rows)
  make_fold_scatter_figure(FIGURES / "figure_16_fold_dependence.svg", folds)
  make_coverage_figure(FIGURES / "figure_17_feature_coverage.svg", group_importance)
  make_confusion_figure(FIGURES / "figure_18_confusion_matrices.svg", research["classification"])
  make_daily_figure(FIGURES / "figure_19_daily_return_paths.svg", daily_rows)
  make_margin_decile_figure(FIGURES / "figure_20_margin_deciles.svg", margin_deciles)

  payload = {
    "probability_diagnostics": probability_rows,
    "fold_correlations": correlation_rows,
    "fold_concentration": concentration_rows,
    "feature_group_importance": group_importance,
    "stage1_horizons": stage1_rows,
    "policy_grid": policy_rows,
    "daily_summary": daily_summary,
    "notes": {
      "feature_importance_source": "split_feature and split_gain arrays parsed from the embedded LightGBM text model in model.pkl",
      "brier_definition": "mean over rows of the sum of three classwise squared probability errors",
      "ece_definition": "10-bin equal-width maximum-probability calibration error",
      "importance_limitation": "split/gain importance is not causal and is biased toward repeatedly splittable variables",
    },
  }
  (REPORT / "technical_results_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(json.dumps({"tables": 13, "supplemental_figures": 11, "output_dir": str(REPORT)}, indent=2))
  return 0


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def build_stage1_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
  metrics = summary["aggregate_metrics"]
  rows = []
  for horizon in (1, 6, 24, 96):
    prefix = f"overall_validation_forward_return_{horizon}_bar"
    rows.append({
      "horizon_bars": horizon,
      "horizon_minutes": horizon * 5,
      "directional_accuracy": metrics[f"{prefix}_directional_accuracy"],
      "mae": metrics[f"{prefix}_mae"],
      "rmse": metrics[f"{prefix}_rmse"],
      "mae_bps": metrics[f"{prefix}_mae"] * 10_000,
      "rmse_bps": metrics[f"{prefix}_rmse"] * 10_000,
      "rows": summary["folds_completed"] * summary["window"]["validation_rows"],
      "folds": summary["folds_completed"],
    })
  return rows


def build_probability_diagnostics(datasets: dict[str, list[dict[str, str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
  summary_rows: list[dict[str, Any]] = []
  calibration_rows: list[dict[str, Any]] = []
  confusion_rows: list[dict[str, Any]] = []
  threshold_rows: list[dict[str, Any]] = []
  classes = (-1, 0, 1)
  columns = {
    -1: "prediction_probability_short",
    0: "prediction_probability_flat",
    1: "prediction_probability_long",
  }
  for dataset, rows in datasets.items():
    targets = [int(float(row["target"])) for row in rows]
    predictions = [int(float(row["prediction"])) for row in rows]
    has_probabilities = all(columns[cls] in rows[0] for cls in classes)
    probabilities = [[float(row[columns[cls]]) for cls in classes] for row in rows] if has_probabilities else []
    margins = [float(row["prediction_margin"]) for row in rows]
    max_probs = [max(values) for values in probabilities]
    confidences = [float(row["prediction_confidence"]) for row in rows] if has_probabilities else []
    correct = [int(a == b) for a, b in zip(targets, predictions)]
    log_loss = -fmean(math.log(max(1e-15, p[classes.index(y)])) for p, y in zip(probabilities, targets)) if has_probabilities else ""
    brier = fmean(sum((p[index] - int(y == cls)) ** 2 for index, cls in enumerate(classes)) for p, y in zip(probabilities, targets)) if has_probabilities else ""
    entropy = fmean(-sum(value * math.log(max(value, 1e-15)) for value in p) / math.log(3) for p in probabilities) if has_probabilities else ""
    bins = []
    for index in range(10) if has_probabilities else ():
      lower, upper = index / 10, (index + 1) / 10
      members = [j for j, value in enumerate(max_probs) if value >= lower and (value < upper or index == 9)]
      if members:
        mean_probability = fmean(max_probs[j] for j in members)
        empirical_accuracy = fmean(correct[j] for j in members)
      else:
        mean_probability = empirical_accuracy = 0.0
      bins.append((members, mean_probability, empirical_accuracy))
      calibration_rows.append({
        "dataset": dataset,
        "bin": index + 1,
        "lower": lower,
        "upper": upper,
        "count": len(members),
        "mean_max_probability": mean_probability,
        "empirical_accuracy": empirical_accuracy,
        "calibration_gap": mean_probability - empirical_accuracy if members else 0.0,
      })
    ece = sum(len(members) / len(rows) * abs(mean_p - accuracy) for members, mean_p, accuracy in bins) if has_probabilities else ""
    quantiles = np.quantile(np.asarray(margins), [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    summary_rows.append({
      "dataset": dataset,
      "rows": len(rows),
      "multiclass_log_loss": log_loss,
      "multiclass_brier_sum": brier,
      "ece_10bin_max_probability": ece,
      "mean_normalized_entropy": entropy,
      "mean_max_probability": fmean(max_probs) if has_probabilities else "",
      "mean_top_two_gap": fmean(confidences) if has_probabilities else "",
      "mean_abs_margin": fmean(abs(value) for value in margins),
      "margin_p01": quantiles[0],
      "margin_p05": quantiles[1],
      "margin_p25": quantiles[2],
      "margin_median": quantiles[3],
      "margin_p75": quantiles[4],
      "margin_p95": quantiles[5],
      "margin_p99": quantiles[6],
      "hard_neutral_share": sum(value == 0 for value in predictions) / len(rows),
      "target_neutral_share": sum(value == 0 for value in targets) / len(rows),
    })
    for threshold in (0.10, 0.12, 0.14, 0.145, 0.15, 0.20):
      threshold_rows.append({
        "dataset": dataset,
        "threshold": threshold,
        "absolute_margin_exceedances": sum(abs(value) > threshold for value in margins),
        "absolute_margin_exceedance_share": sum(abs(value) > threshold for value in margins) / len(rows),
        "positive_margin_exceedances": sum(value > threshold for value in margins),
        "negative_margin_exceedances": sum(value < -threshold for value in margins),
      })
    for actual_index, actual in enumerate(classes):
      for predicted_index, predicted in enumerate(classes):
        count = sum(a == actual and p == predicted for a, p in zip(targets, predictions))
        confusion_rows.append({
          "dataset": dataset,
          "actual_class": actual,
          "predicted_class": predicted,
          "count": count,
          "row_share": count / max(1, sum(a == actual for a in targets)),
          "sample_share": count / len(rows),
          "actual_index": actual_index,
          "predicted_index": predicted_index,
        })
  return summary_rows, calibration_rows, confusion_rows, threshold_rows


def feature_group(name: str) -> str:
  if name.startswith("oof_"):
    return "upstream_quantiles"
  if name.startswith("structure_supply_zone") or name.startswith("structure_demand_zone"):
    return "supply_demand_zones"
  if name.startswith("structure_"):
    return "generic_structure"
  if name.startswith("trend_"):
    return "trend_path"
  if name.startswith("regime_"):
    return "regime_change"
  if name.startswith("trades_"):
    return "trade_window"
  if name.startswith("top_"):
    return "order_book_top"
  return "price_bar"


def feature_completeness(feature_names: list[str]) -> dict[str, dict[str, Any]]:
  stats = {
    name: {"blank": 0, "zero": 0, "nonzero": 0, "n": 0, "mean": 0.0, "m2": 0.0, "min": math.inf, "max": -math.inf}
    for name in feature_names
  }
  rows = 0
  for path in sorted(FEATURE_ROOT.rglob("*.csv")):
    for row in read_csv(path):
      rows += 1
      for name in feature_names:
        raw = row.get(name, "")
        if raw == "" or raw is None:
          stats[name]["blank"] += 1
          value = 0.0
        else:
          value = float(raw)
        if abs(value) <= 1e-15:
          stats[name]["zero"] += 1
        else:
          stats[name]["nonzero"] += 1
        current = stats[name]
        current["n"] += 1
        delta = value - current["mean"]
        current["mean"] += delta / current["n"]
        current["m2"] += delta * (value - current["mean"])
        current["min"] = min(current["min"], value)
        current["max"] = max(current["max"], value)
  for values in stats.values():
    values["rows"] = rows
    values["blank_rate"] = values["blank"] / rows
    values["zero_rate_after_imputation"] = values["zero"] / rows
    values["std_after_imputation"] = math.sqrt(values["m2"] / rows)
  return stats


def feature_importance(feature_names: list[str], model_path: Path, completeness: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  text = model_path.read_bytes().decode("latin1")
  matches = re.findall(r"split_feature=([0-9 ]+)\nsplit_gain=([0-9eE+.\- ]+)", text)
  split_counts: Counter[int] = Counter()
  gains: defaultdict[int, float] = defaultdict(float)
  for feature_text, gain_text in matches:
    features = [int(value) for value in feature_text.split()]
    values = [float(value) for value in gain_text.split()]
    for index, gain in zip(features, values):
      split_counts[index] += 1
      gains[index] += gain
  total_splits = sum(split_counts.values())
  total_gain = sum(gains.values())
  rows = []
  for index, name in enumerate(feature_names):
    coverage = completeness[name]
    rows.append({
      "feature_index": index,
      "feature": name,
      "group": feature_group(name),
      "split_count": split_counts[index],
      "split_share": split_counts[index] / max(1, total_splits),
      "gain": gains[index],
      "gain_share": gains[index] / max(1e-15, total_gain),
      "blank_count": coverage["blank"],
      "blank_rate": coverage["blank_rate"],
      "zero_count_after_imputation": coverage["zero"],
      "zero_rate_after_imputation": coverage["zero_rate_after_imputation"],
      "nonzero_count": coverage["nonzero"],
      "mean_after_imputation": coverage["mean"],
      "std_after_imputation": coverage["std_after_imputation"],
      "minimum_after_imputation": coverage["min"],
      "maximum_after_imputation": coverage["max"],
    })
  rows.sort(key=lambda row: (-row["gain"], -row["split_count"], row["feature_index"]))
  for rank, row in enumerate(rows, start=1):
    row["gain_rank"] = rank
  return rows


def aggregate_group_importance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for row in rows:
    groups[row["group"]].append(row)
  result = []
  for name, members in groups.items():
    result.append({
      "group": name,
      "feature_count": len(members),
      "used_feature_count": sum(row["split_count"] > 0 for row in members),
      "split_count": sum(row["split_count"] for row in members),
      "split_share": sum(row["split_share"] for row in members),
      "gain": sum(row["gain"] for row in members),
      "gain_share": sum(row["gain_share"] for row in members),
      "mean_blank_rate": fmean(row["blank_rate"] for row in members),
      "mean_zero_rate_after_imputation": fmean(row["zero_rate_after_imputation"] for row in members),
    })
  return sorted(result, key=lambda row: -row["gain"])


def pearson(x: list[float], y: list[float]) -> float:
  if len(x) < 2:
    return 0.0
  x_mean, y_mean = fmean(x), fmean(y)
  numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
  denominator = math.sqrt(sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y))
  return numerator / denominator if denominator else 0.0


def ranks(values: list[float]) -> list[float]:
  order = sorted(range(len(values)), key=lambda index: values[index])
  result = [0.0] * len(values)
  start = 0
  while start < len(order):
    end = start + 1
    while end < len(order) and values[order[end]] == values[order[start]]:
      end += 1
    rank = (start + 1 + end) / 2
    for index in order[start:end]:
      result[index] = rank
    start = end
  return result


def fold_diagnostics(folds: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  measures = {
    "accuracy": [float(row["validation_accuracy"]) for row in folds],
    "balanced_accuracy": [float(row["validation_macro_recall"]) for row in folds],
    "temperature": [float(row["calibration_temperature"]) for row in folds],
    "original_entries": [float(row["original_entries"]) for row in folds],
    "selected_entries": [float(row["selected_entries"]) for row in folds],
  }
  returns = {
    "original_return": [float(row["original_net_return"]) for row in folds],
    "selected_return": [float(row["selected_net_return"]) for row in folds],
  }
  correlations = []
  for x_name, x in measures.items():
    for y_name, y in returns.items():
      correlations.append({
        "x": x_name,
        "y": y_name,
        "pearson": pearson(x, y),
        "spearman": pearson(ranks(x), ranks(y)),
        "folds": len(folds),
      })
  concentration = []
  for policy, column in (("original", "original_net_return"), ("selected", "selected_net_return")):
    values = [float(row[column]) for row in folds]
    positive = [max(0.0, value) for value in values]
    total_positive = sum(positive)
    shares = [value / total_positive for value in positive if value > 0]
    hhi = sum(value * value for value in shares)
    aggregate = math.prod(1 + value for value in values) - 1
    leave_one_out = [math.prod(1 + value for j, value in enumerate(values) if j != index) - 1 for index in range(len(values))]
    concentration.append({
      "policy": policy,
      "folds": len(values),
      "positive_folds": sum(value > 0 for value in values),
      "zero_folds": sum(value == 0 for value in values),
      "median_fold_return": median(values),
      "mean_fold_return": fmean(values),
      "fold_return_std_population": float(np.std(np.asarray(values), ddof=0)),
      "best_fold_return": max(values),
      "worst_fold_return": min(values),
      "aggregate_compounded_return": aggregate,
      "best_positive_fold_share": max(shares) if shares else 0.0,
      "positive_profit_hhi": hhi,
      "effective_positive_fold_count": 1 / hhi if hhi else 0.0,
      "minimum_leave_one_fold_out_return": min(leave_one_out),
      "maximum_leave_one_fold_out_return": max(leave_one_out),
    })
  return correlations, concentration


def policy_grid_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for rank, candidate in enumerate(summary["top_candidates"], start=1):
    rows.append({
      "selection_rank": rank,
      "variant": candidate["variant_slug"],
      "margin_threshold": candidate["margin_threshold"],
      "minimum_hold_bars": candidate["minimum_hold_bars"],
      "minimum_rebalance_delta": candidate["minimum_rebalance_delta"],
      "fixed_target_position": candidate["fixed_target_position"],
      "net_return": candidate["net_return"],
      "stress_net_return_3bps": candidate.get("stress_net_return", ""),
      "max_drawdown": candidate["max_drawdown"],
      "entries": candidate["entries"],
      "position_adjustments": candidate["position_adjustments"],
      "turnover": candidate["total_turnover"],
      "positive_folds": candidate["positive_folds"],
      "traded_folds": candidate["traded_folds"],
      "median_fold_return": candidate["median_fold_return"],
      "best_positive_fold_share": candidate["best_fold_profit_share"],
      "net_edge_per_turnover_bps": candidate["net_edge_per_turnover_bps"],
      "feasible": candidate["feasible"],
      "requirement_failures": ";".join(candidate["requirement_failures"]),
    })
  return rows


def daily_diagnostics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
  by_run: dict[str, list[float]] = defaultdict(list)
  for row in rows:
    by_run[row["run"]].append(float(row["net_return"]))
  result = []
  for run, values in by_run.items():
    positive = [max(0, value) for value in values]
    positive_total = sum(positive)
    result.append({
      "run": run,
      "days": len(values),
      "positive_days": sum(value > 0 for value in values),
      "zero_days": sum(value == 0 for value in values),
      "negative_days": sum(value < 0 for value in values),
      "positive_day_share": sum(value > 0 for value in values) / len(values),
      "mean_daily_return": fmean(values),
      "median_daily_return": median(values),
      "daily_std_sample": float(np.std(np.asarray(values), ddof=1)) if len(values) > 1 else 0.0,
      "best_day": max(values),
      "worst_day": min(values),
      "compounded_return": math.prod(1 + value for value in values) - 1,
      "best_day_share_of_positive_sum": max(positive) / positive_total if positive_total else 0.0,
    })
  return result


def build_margin_deciles(datasets: dict[str, list[dict[str, str]]], periods: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
  output = []
  for dataset, prediction_rows in datasets.items():
    return_by_time = {row["candle_start"]: float(row["realized_return"]) for row in periods[dataset]}
    joined = [
      (float(row["prediction_margin"]), return_by_time[row["candle_start"]])
      for row in prediction_rows
      if row["candle_start"] in return_by_time
    ]
    joined.sort(key=lambda item: item[0])
    for decile in range(10):
      lower = decile * len(joined) // 10
      upper = (decile + 1) * len(joined) // 10
      subset = joined[lower:upper]
      margins = [item[0] for item in subset]
      returns = [item[1] for item in subset]
      output.append({
        "dataset": dataset,
        "margin_decile": decile + 1,
        "rows": len(subset),
        "minimum_margin": min(margins),
        "maximum_margin": max(margins),
        "mean_margin": fmean(margins),
        "mean_forward_return": fmean(returns),
        "median_forward_return": median(returns),
        "positive_forward_return_share": sum(value > 0 for value in returns) / len(returns),
        "margin_return_pearson_within_bin": pearson(margins, returns),
      })
  return output


def build_split_rows(research: dict[str, Any]) -> list[dict[str, Any]]:
  metadata = research["artifact_metadata"]
  calibration = metadata["extra"]["probability_calibration"]
  protocol = research["oof_evaluation"]["protocol"]
  return [
    {"partition": "stage1 complete sequence table", "rows": 25760, "start": "2026-01-16T07:45:00Z", "end": "2026-04-15T18:25:00Z", "status": "all eligible sequence endpoints"},
    {"partition": "stage1 expanding OOF predictions", "rows": 23552, "start": "2026-01-23T10:30:00Z", "end": "2026-04-15T05:10:00Z", "status": "46 fold validation blocks; validation-tuned checkpoints"},
    {"partition": "final LightGBM fit", "rows": calibration["fit_rows"], "start": "2026-01-23T10:30:00Z", "end": "2026-03-20T01:10:00Z", "status": "tree fitting"},
    {"partition": "final temperature calibration", "rows": calibration["calibration_rows"], "start": "2026-03-20T01:15:00Z", "end": "2026-03-29T20:40:00Z", "status": "temperature selected by NLL"},
    {"partition": "final model validation", "rows": metadata["validation_rows"], "start": "2026-03-29T20:45:00Z", "end": "2026-04-15T05:10:00Z", "status": "development evidence"},
    {"partition": "meta expanding OOF", "rows": protocol["oof_rows"], "start": protocol["oof_start"], "end": protocol["oof_end"], "status": "14 disjoint 1024-row blocks; purge=1, embargo=1"},
    {"partition": "later policy replay", "rows": research["forward_evaluation"]["periods"], "start": research["forward_evaluation"]["evaluation_start_time"], "end": research["forward_evaluation"]["evaluation_end_time"], "status": "parameter-OOS, previously inspected"},
  ]


def canvas(title: str, subtitle: str) -> list[str]:
  return paper.svg_canvas(title, subtitle)


def make_stage1_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 10. Upstream quantile-forecast diagnostics by horizon", "Directional accuracy is near 0.50; absolute errors increase with the forecast horizon")
  labels = [f"{row['horizon_bars']} bars" for row in rows]
  paper.bar_panel(elements, 70, 135, 500, 400, labels, [row["directional_accuracy"] for row in rows], [COLORS["blue"]] * 4, "Directional accuracy", "decimal")
  paper.grouped_bar_panel(elements, 650, 135, 480, 400, labels, [("MAE", [row["mae_bps"] for row in rows], COLORS["teal"]), ("RMSE", [row["rmse_bps"] for row in rows], COLORS["amber"])], "Forecast error (bps)", "decimal")
  paper.add_legend(elements, 450, 620, [("MAE", COLORS["teal"]), ("RMSE", COLORS["amber"])])
  paper.save_svg(path, elements)


def make_reliability_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 11. Maximum-probability reliability", "Ten equal-width bins; forward replay omitted because its saved output contains hard classes and margins but not class probabilities")
  for panel, dataset in enumerate(("walk_forward_oof", "final_model_validation")):
    current = [row for row in rows if row["dataset"] == dataset and int(row["count"]) > 0]
    x_values = [float(row["mean_max_probability"]) for row in current]
    accuracies = [float(row["empirical_accuracy"]) for row in current]
    x = 75 + panel * 580
    paper.line_panel(elements, x, 150, 500, 360, [("observed", accuracies, [COLORS["blue"], COLORS["teal"]][panel]), ("ideal", x_values, COLORS["gray"])], dataset.replace("_", " "), percent_axis=True, include_zero=True, x_values=x_values, x_label="mean maximum probability", y_fixed=(0, 1))
  paper.add_legend(elements, 440, 620, [("Observed accuracy", COLORS["blue"]), ("Ideal", COLORS["gray"])])
  paper.save_svg(path, elements)


def make_importance_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 12. Final LightGBM feature importance", "Top 20 by summed split gain recovered from the serialized booster; importance is not causal attribution")
  maximum = max(row["gain_share"] for row in rows)
  x, y, width, line_height = 470, 110, 650, 25
  for index, row in enumerate(rows):
    py = y + index * line_height
    bar = row["gain_share"] / maximum * width
    elements.append(f"<text x='{x - 12}' y='{py + 16}' text-anchor='end' font-family='Arial,sans-serif' font-size='11' fill='{COLORS['ink']}'>{escape(row['feature'])}</text>")
    elements.append(f"<rect x='{x}' y='{py + 4}' width='{bar:.1f}' height='16' rx='2' fill='{COLORS['blue']}'/>")
    elements.append(f"<text x='{x + bar + 6:.1f}' y='{py + 16}' font-family='Arial,sans-serif' font-size='10' fill='{COLORS['gray']}'>{row['gain_share'] * 100:.2f}%</text>")
  paper.save_svg(path, elements)


def make_group_importance_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 13. Importance aggregated by feature family", "Gain and split shares sum to one; feature counts differ substantially by family")
  labels = [row["group"].replace("_", " ") for row in rows]
  paper.grouped_bar_panel(elements, 80, 135, 1040, 400, labels, [("gain share", [row["gain_share"] for row in rows], COLORS["blue"]), ("split share", [row["split_share"] for row in rows], COLORS["teal"])], "Share of serialized tree importance", "percent")
  paper.add_legend(elements, 440, 620, [("Gain share", COLORS["blue"]), ("Split share", COLORS["teal"])])
  paper.save_svg(path, elements)


def make_margin_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 14. Probability-margin exceedance rates", "Raw |P(long)-P(short)| exceedance; downstream trend filtering and state constraints reduce actual entries")
  thresholds = (0.10, 0.12, 0.15, 0.20)
  labels = [f"{value:.2f}" for value in thresholds]
  series = []
  for dataset, label, color in (("walk_forward_oof", "OOF", COLORS["blue"]), ("final_model_validation", "Validation", COLORS["teal"]), ("forward_policy_replay", "Replay", COLORS["red"])):
    lookup = {(row["dataset"], float(row["threshold"])): float(row["absolute_margin_exceedance_share"]) for row in rows}
    series.append((label, [lookup[(dataset, value)] for value in thresholds], color))
  paper.grouped_bar_panel(elements, 100, 135, 1000, 400, labels, series, "Share of rows exceeding absolute margin threshold", "percent")
  paper.add_legend(elements, 420, 620, [(name, color) for name, _, color in series])
  paper.save_svg(path, elements)


def make_policy_grid_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 15. Post-hoc execution-policy search surface", "Nine OOF variants; green ring marks the sole feasible variant, while the raw-return maximum is infeasible")
  x0, y0, width, height = 100, 130, 1000, 420
  values = [float(row["net_return"]) for row in rows]
  y_min, y_max = min(values) * 0.9, max(values) * 1.08
  paper.draw_axes(elements, x0, y0, width, height, y_min, y_max, percent_axis=True)
  thresholds = sorted({float(row["margin_threshold"]) for row in rows})
  for row in rows:
    threshold = float(row["margin_threshold"])
    hold = int(row["minimum_hold_bars"])
    px = x0 + (threshold - min(thresholds)) / (max(thresholds) - min(thresholds)) * width
    px += (hold - 9) * 22
    value = float(row["net_return"])
    py = y0 + height - (value - y_min) / (y_max - y_min) * height
    color = {8: COLORS["amber"], 9: COLORS["blue"], 10: COLORS["teal"]}[hold]
    elements.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='10' fill='{color}' stroke='{COLORS['green'] if row['feasible'] else COLORS['paper']}' stroke-width='4'/>")
    elements.append(f"<text x='{px:.1f}' y='{py - 15:.1f}' text-anchor='middle' font-family='Arial,sans-serif' font-size='10' fill='{COLORS['ink']}'>{value * 100:.2f}%</text>")
  for threshold in thresholds:
    px = x0 + (threshold - min(thresholds)) / (max(thresholds) - min(thresholds)) * width
    elements.append(f"<text x='{px:.1f}' y='{y0 + height + 28}' text-anchor='middle' font-family='Arial,sans-serif' font-size='12' fill='{COLORS['gray']}'>{threshold:.3f}</text>")
  paper.add_legend(elements, 380, 620, [("hold 8", COLORS["amber"]), ("hold 9", COLORS["blue"]), ("hold 10", COLORS["teal"]), ("feasible ring", COLORS["green"])])
  paper.save_svg(path, elements)


def make_fold_scatter_figure(path: Path, folds: list[dict[str, str]]) -> None:
  elements = canvas("Figure 16. Fold-level predictive scores versus realized return", "Each point is one 1,024-row meta OOF fold; labels identify fold number")
  panels = [("Original policy", "original_net_return", COLORS["blue"]), ("Selected policy", "selected_net_return", COLORS["teal"])]
  for panel, (title, return_column, color) in enumerate(panels):
    x0, y0, width, height = 80 + panel * 575, 140, 490, 390
    x_values = [float(row["validation_macro_recall"]) for row in folds]
    y_values = [float(row[return_column]) for row in folds]
    x_min, x_max = min(x_values) - 0.005, max(x_values) + 0.005
    y_min, y_max = paper.padded_range(y_values, include_zero=True)
    paper.draw_axes(elements, x0, y0, width, height, y_min, y_max, percent_axis=True)
    elements.append(f"<text x='{x0}' y='{y0 - 18}' font-family='Arial,sans-serif' font-size='17' font-weight='700' fill='{COLORS['navy']}'>{title}</text>")
    for row, x_value, y_value in zip(folds, x_values, y_values):
      px = x0 + (x_value - x_min) / (x_max - x_min) * width
      py = y0 + height - (y_value - y_min) / (y_max - y_min) * height
      elements.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='6' fill='{color}' opacity='.85'/>")
      elements.append(f"<text x='{px + 7:.1f}' y='{py - 5:.1f}' font-family='Arial,sans-serif' font-size='9' fill='{COLORS['gray']}'>F{row['fold_index']}</text>")
    elements.append(f"<text x='{x0 + width / 2}' y='{y0 + height + 42}' text-anchor='middle' font-family='Arial,sans-serif' font-size='11' fill='{COLORS['gray']}'>balanced accuracy</text>")
  paper.save_svg(path, elements)


def make_coverage_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 17. Realized feature coverage and tree usage", "Zero rates are measured after the training loader's blank-to-zero transformation")
  ordered = sorted(rows, key=lambda row: row["group"])
  x0, y0, width, row_height = 410, 125, 650, 55
  for index, row in enumerate(ordered):
    py = y0 + index * row_height
    coverage = 1 - float(row["mean_zero_rate_after_imputation"])
    used = row["used_feature_count"] / row["feature_count"]
    elements.append(f"<text x='{x0 - 15}' y='{py + 15}' text-anchor='end' font-family='Arial,sans-serif' font-size='12' fill='{COLORS['ink']}'>{escape(row['group'].replace('_', ' '))}</text>")
    elements.append(f"<rect x='{x0}' y='{py}' width='{width}' height='15' fill='{COLORS['light']}'/>")
    elements.append(f"<rect x='{x0}' y='{py}' width='{coverage * width:.1f}' height='15' fill='{COLORS['blue']}'/>")
    elements.append(f"<rect x='{x0}' y='{py + 22}' width='{width}' height='15' fill='{COLORS['light']}'/>")
    elements.append(f"<rect x='{x0}' y='{py + 22}' width='{used * width:.1f}' height='15' fill='{COLORS['teal']}'/>")
    elements.append(f"<text x='{x0 + width + 8}' y='{py + 13}' font-family='Arial,sans-serif' font-size='10' fill='{COLORS['gray']}'>{coverage * 100:.1f}% nonzero</text>")
    elements.append(f"<text x='{x0 + width + 8}' y='{py + 35}' font-family='Arial,sans-serif' font-size='10' fill='{COLORS['gray']}'>{row['used_feature_count']}/{row['feature_count']} used</text>")
  paper.add_legend(elements, 440, 630, [("Nonzero observation rate", COLORS["blue"]), ("Feature used in a split", COLORS["teal"])])
  paper.save_svg(path, elements)


def make_confusion_figure(path: Path, classifications: dict[str, Any]) -> None:
  elements = canvas("Figure 18. Complete confusion matrices", "Rows are actual classes and columns predicted classes; counts are unnormalized")
  specs = [("walk_forward_oof", "Meta OOF"), ("final_model_validation", "Final validation"), ("forward_policy_holdout", "Later replay")]
  for index, (key, title) in enumerate(specs):
    paper.confusion_panel(elements, 80 + index * 385, 165, 270, classifications[key]["confusion_matrix"], ["−1", "0", "+1"], title)
  paper.save_svg(path, elements)


def make_daily_figure(path: Path, rows: list[dict[str, str]]) -> None:
  elements = canvas("Figure 19. Compounded daily-return paths", "Daily aggregation of bar-level net returns; forward samples contain only six daily observations")
  by_run: dict[str, list[float]] = defaultdict(list)
  for row in rows:
    by_run[row["run"]].append(float(row["net_return"]))
  specs = [("OOF", "oof_original", "oof_selected"), ("Forward replay", "forward_original", "forward_selected")]
  for panel, (title, original, selected) in enumerate(specs):
    series = []
    for label, key, color in (("Original", original, COLORS["blue"]), ("Selected", selected, COLORS["teal"])):
      values = [1.0]
      for value in by_run[key]:
        values.append(values[-1] * (1 + value))
      series.append((label, values, color))
    paper.line_panel(elements, 70 + panel * 580, 130, 520, 400, series, title, percent_axis=False, include_zero=False, x_label="elapsed day")
  paper.add_legend(elements, 430, 620, [("Original", COLORS["blue"]), ("Selected", COLORS["teal"])])
  paper.save_svg(path, elements)


def make_margin_decile_figure(path: Path, rows: list[dict[str, Any]]) -> None:
  elements = canvas("Figure 20. Forward return by probability-margin decile", "Unconditional next-bar return within equal-count bins of P(long)-P(short); this is not the filtered policy return")
  labels = [f"D{value}" for value in range(1, 11)]
  series = []
  for dataset, label, color in (("walk_forward_oof", "OOF", COLORS["blue"]), ("final_model_validation", "Validation", COLORS["teal"]), ("forward_policy_replay", "Replay", COLORS["red"])):
    current = sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: row["margin_decile"])
    series.append((label, [row["mean_forward_return"] for row in current], color))
  paper.grouped_bar_panel(elements, 80, 135, 1040, 400, labels, series, "Mean next-bar return", "percent")
  paper.add_legend(elements, 420, 620, [(name, color) for name, _, color in series])
  paper.save_svg(path, elements)


def escape(value: str) -> str:
  return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
  raise SystemExit(main())
