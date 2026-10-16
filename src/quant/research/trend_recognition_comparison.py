from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant.models.base import ModelDependencyError, utc_run_slug
from quant.models.probability_calibration import TemperatureScaler


def _default_optimization_weights() -> dict[str, float]:
  return {
    "transition.macro_recall": 0.3,
    "macro_recall": 0.2,
    "recall.0": 0.15,
    "transition.recall.0": 0.15,
    "accuracy": 0.1,
    "trend_direction_accuracy_ex_chop": 0.1,
  }


@dataclass(frozen=True, slots=True)
class TrendRecognitionComparisonConfig:
  feature_root: Path
  baseline_prediction_path: Path
  trend_prediction_path: Path
  output_root: Path = Path("reports/trend_recognition")
  provider: str = "coinbase"
  symbol: str = "BTC-USD"
  dataset_name: str = "market_supervised"
  target_column: str = "future_trend_state_label_24_bar"
  regime_change_column: str = "regime_bocpd_change_prob"
  regime_change_threshold: float = 0.35
  baseline_return_prediction_column: str = "prediction_return_q50_24_bar"
  trend_return_prediction_column: str = "prediction_return_q50_24_bar"
  trend_state_prediction_column: str = "prediction_trend_state_24_bar"
  uptrend_probability_column: str = "prediction_probability_uptrend_24_bar"
  downtrend_probability_column: str = "prediction_probability_downtrend_24_bar"
  chop_probability_column: str = "prediction_probability_chop_24_bar"
  gate_thresholds: tuple[float, ...] = (0.55, 0.6, 0.65)
  chop_thresholds: tuple[float, ...] = (0.55, 0.6, 0.7)
  countertrend_scales: tuple[float, ...] = (0.0, 0.15, 0.25, 0.5)
  calibration_fraction: float = 0.5
  minimum_calibration_rows: int = 128
  probability_calibration_method: str = "none"
  optimization_weights: dict[str, float] = field(default_factory=_default_optimization_weights)
  max_accuracy_drop: float = 0.03
  max_trend_direction_accuracy_drop: float = 0.075
  constraint_penalty_weight: float = 2.0


@dataclass(frozen=True, slots=True)
class TrendRecognitionComparisonSummary:
  output_path: Path
  common_rows: int
  best_profile: dict[str, Any] | None
  calibration_rows: int = 0
  holdout_rows: int = 0
  calibrated_holdout_profile: dict[str, Any] | None = None


def run_trend_recognition_comparison(
  config: TrendRecognitionComparisonConfig,
) -> TrendRecognitionComparisonSummary:
  labels = _load_feature_labels(config)
  baseline_rows = _load_prediction_rows(config.baseline_prediction_path)
  trend_rows = _load_prediction_rows(config.trend_prediction_path)
  baseline_by_time = {row.get("candle_start", ""): row for row in baseline_rows}
  trend_by_time = {row.get("candle_start", ""): row for row in trend_rows}
  common_times = sorted(set(baseline_by_time) & set(trend_by_time))

  baseline_all = _evaluate_rows(
    rows=baseline_rows,
    labels=labels,
    config=config,
    prediction_fn=lambda row: _sign(_float(row, config.baseline_return_prediction_column)),
  )
  trend_aux_all = _evaluate_rows(
    rows=trend_rows,
    labels=labels,
    config=config,
    prediction_fn=lambda row: int(round(_float(row, config.trend_state_prediction_column))),
  )
  trend_return_all = _evaluate_rows(
    rows=trend_rows,
    labels=labels,
    config=config,
    prediction_fn=lambda row: _sign(_float(row, config.trend_return_prediction_column)),
  )
  baseline_common = _evaluate_times(
    common_times,
    labels=labels,
    config=config,
    row_by_time=baseline_by_time,
    prediction_fn=lambda row: _sign(_float(row, config.baseline_return_prediction_column)),
  )
  trend_aux_common = _evaluate_times(
    common_times,
    labels=labels,
    config=config,
    row_by_time=trend_by_time,
    prediction_fn=lambda row: int(round(_float(row, config.trend_state_prediction_column))),
  )
  trend_return_common = _evaluate_times(
    common_times,
    labels=labels,
    config=config,
    row_by_time=trend_by_time,
    prediction_fn=lambda row: _sign(_float(row, config.trend_return_prediction_column)),
  )
  gate_sweep = _gate_sweep(
    common_times,
    labels=labels,
    baseline_by_time=baseline_by_time,
    trend_by_time=trend_by_time,
    config=config,
    baseline_metrics=baseline_common,
  )
  best_profile = max(gate_sweep, key=lambda row: row["optimization_score"], default=None)
  calibration_times, holdout_times = _calibration_holdout_times(common_times, config=config)
  baseline_calibration = _evaluate_times(
    calibration_times,
    labels=labels,
    config=config,
    row_by_time=baseline_by_time,
    prediction_fn=lambda row: _sign(_float(row, config.baseline_return_prediction_column)),
  ) if calibration_times else {}
  baseline_holdout = _evaluate_times(
    holdout_times,
    labels=labels,
    config=config,
    row_by_time=baseline_by_time,
    prediction_fn=lambda row: _sign(_float(row, config.baseline_return_prediction_column)),
  ) if holdout_times else {}
  probability_calibrator, probability_calibration = _fit_probability_calibrator(
    calibration_times,
    labels=labels,
    trend_by_time=trend_by_time,
    config=config,
  )
  calibrated_probabilities_by_time = _probabilities_by_time(
    common_times,
    trend_by_time=trend_by_time,
    config=config,
    probability_calibrator=probability_calibrator,
  )
  calibration_gate_sweep = _gate_sweep(
    calibration_times,
    labels=labels,
    baseline_by_time=baseline_by_time,
    trend_by_time=trend_by_time,
    config=config,
    probabilities_by_time=calibrated_probabilities_by_time,
    baseline_metrics=baseline_calibration,
  )
  calibrated_holdout_profile = _calibrated_holdout_profile(
    calibration_gate_sweep,
    holdout_times,
    labels=labels,
    baseline_by_time=baseline_by_time,
    trend_by_time=trend_by_time,
    config=config,
    probabilities_by_time=calibrated_probabilities_by_time,
    baseline_metrics=baseline_holdout,
  )

  payload = {
    "feature_root": str(config.feature_root),
    "baseline_prediction_path": str(config.baseline_prediction_path),
    "trend_prediction_path": str(config.trend_prediction_path),
    "target_column": config.target_column,
    "regime_change_column": config.regime_change_column,
    "regime_change_threshold": config.regime_change_threshold,
    "common_rows": len(common_times),
    "baseline_all": baseline_all,
    "trend_aware_aux_all": trend_aux_all,
    "trend_aware_return_sign_all": trend_return_all,
    "baseline_common": baseline_common,
    "trend_aware_aux_common": trend_aux_common,
    "trend_aware_return_sign_common": trend_return_common,
    "trend_probability_calibration_common": _probability_calibration_metrics(
      common_times,
      labels=labels,
      trend_by_time=trend_by_time,
      config=config,
    ),
    "gate_sweep": gate_sweep,
    "best_profile": best_profile,
    "calibration": {
      "calibration_fraction": config.calibration_fraction,
      "minimum_calibration_rows": config.minimum_calibration_rows,
      "calibration_rows": len(calibration_times),
      "holdout_rows": len(holdout_times),
      "probability_calibration": probability_calibration,
      "optimization_weights": config.optimization_weights,
      "max_accuracy_drop": config.max_accuracy_drop,
      "max_trend_direction_accuracy_drop": config.max_trend_direction_accuracy_drop,
      "constraint_penalty_weight": config.constraint_penalty_weight,
      "baseline_calibration": baseline_calibration,
      "baseline_holdout": baseline_holdout,
      "trend_aware_aux_holdout": _evaluate_times(
        holdout_times,
        labels=labels,
        config=config,
        row_by_time=trend_by_time,
        prediction_fn=lambda row: int(round(_float(row, config.trend_state_prediction_column))),
      ) if holdout_times else {},
      "raw_probability_calibration_holdout": _probability_calibration_metrics(
        holdout_times,
        labels=labels,
        trend_by_time=trend_by_time,
        config=config,
      ) if holdout_times else {},
      "calibrated_probability_calibration_holdout": _probability_calibration_metrics(
        holdout_times,
        labels=labels,
        trend_by_time=trend_by_time,
        config=config,
        probability_calibrator=probability_calibrator,
      ) if holdout_times else {},
      "gate_sweep": calibration_gate_sweep,
      "selected_holdout_profile": calibrated_holdout_profile,
    },
  }
  output_path = config.output_root / f"{utc_run_slug()}.json"
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  return TrendRecognitionComparisonSummary(
    output_path=output_path,
    common_rows=len(common_times),
    best_profile=best_profile,
    calibration_rows=len(calibration_times),
    holdout_rows=len(holdout_times),
    calibrated_holdout_profile=calibrated_holdout_profile,
  )


def _load_feature_labels(config: TrendRecognitionComparisonConfig) -> dict[str, dict[str, str]]:
  root = config.feature_root / "training_datasets" / config.dataset_name / config.provider / config.symbol
  rows: dict[str, dict[str, str]] = {}
  for path in sorted(root.rglob("*.csv")):
    with path.open("r", encoding="utf-8", newline="") as handle:
      for row in csv.DictReader(handle):
        rows[row.get("candle_start", "")] = row
  return rows


def _load_prediction_rows(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def _evaluate_rows(
  *,
  rows: list[dict[str, str]],
  labels: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  prediction_fn: Any,
) -> dict[str, Any]:
  return _trend_metrics(
    _pairs_for_rows(rows, labels=labels, config=config, prediction_fn=prediction_fn),
    transition_pairs=_pairs_for_rows(rows, labels=labels, config=config, prediction_fn=prediction_fn, transitions_only=True),
  )


def _evaluate_times(
  times: list[str],
  *,
  labels: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  row_by_time: dict[str, dict[str, str]],
  prediction_fn: Any,
) -> dict[str, Any]:
  rows = [row_by_time[time] for time in times]
  return _evaluate_rows(rows=rows, labels=labels, config=config, prediction_fn=prediction_fn)


def _pairs_for_rows(
  rows: list[dict[str, str]],
  *,
  labels: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  prediction_fn: Any,
  transitions_only: bool = False,
) -> list[tuple[int, int]]:
  pairs: list[tuple[int, int]] = []
  for row in rows:
    label_row = labels.get(row.get("candle_start", ""), {})
    target_value = label_row.get(config.target_column, "")
    if target_value == "":
      continue
    if transitions_only and _float(label_row, config.regime_change_column) < config.regime_change_threshold:
      continue
    pairs.append((int(float(target_value)), int(prediction_fn(row))))
  return pairs


def _gate_sweep(
  common_times: list[str],
  *,
  labels: dict[str, dict[str, str]],
  baseline_by_time: dict[str, dict[str, str]],
  trend_by_time: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  probabilities_by_time: dict[str, tuple[float, float, float]] | None = None,
  baseline_metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
  output: list[dict[str, Any]] = []
  for gate_threshold in config.gate_thresholds:
    for chop_threshold in config.chop_thresholds:
      for countertrend_scale in config.countertrend_scales:
        metrics = _profile_metrics(
          common_times,
          labels=labels,
          baseline_by_time=baseline_by_time,
          trend_by_time=trend_by_time,
          config=config,
          gate_threshold=gate_threshold,
          chop_threshold=chop_threshold,
          countertrend_scale=countertrend_scale,
          probabilities_by_time=probabilities_by_time,
        )
        profile = {
          "trend_state_gate_threshold": gate_threshold,
          "trend_state_chop_threshold": chop_threshold,
          "trend_state_countertrend_scale": countertrend_scale,
          "metrics": metrics,
          "optimization_score": _optimization_score(metrics, config=config, baseline_metrics=baseline_metrics),
        }
        output.append(profile)
  return sorted(output, key=lambda row: row["optimization_score"], reverse=True)


def _profile_metrics(
  times: list[str],
  *,
  labels: dict[str, dict[str, str]],
  baseline_by_time: dict[str, dict[str, str]],
  trend_by_time: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  gate_threshold: float,
  chop_threshold: float,
  countertrend_scale: float,
  probabilities_by_time: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
  pairs: list[tuple[int, int]] = []
  transition_pairs: list[tuple[int, int]] = []
  for time in times:
    label_row = labels.get(time, {})
    target_value = label_row.get(config.target_column, "")
    if target_value == "":
      continue
    base_direction = _sign(_float(baseline_by_time[time], config.baseline_return_prediction_column))
    gated_direction = _gated_direction(
      base_direction=base_direction,
      trend_row=trend_by_time[time],
      config=config,
      gate_threshold=gate_threshold,
      chop_threshold=chop_threshold,
      countertrend_scale=countertrend_scale,
      probabilities=probabilities_by_time.get(time) if probabilities_by_time else None,
    )
    pair = (int(float(target_value)), gated_direction)
    pairs.append(pair)
    if _float(label_row, config.regime_change_column) >= config.regime_change_threshold:
      transition_pairs.append(pair)
  return _trend_metrics(pairs, transition_pairs=transition_pairs)


def _gated_direction(
  *,
  base_direction: int,
  trend_row: dict[str, str],
  config: TrendRecognitionComparisonConfig,
  gate_threshold: float,
  chop_threshold: float,
  countertrend_scale: float,
  probabilities: tuple[float, float, float] | None = None,
) -> int:
  downtrend_probability, chop_probability, uptrend_probability = probabilities or _trend_probabilities(trend_row, config)
  if chop_probability >= chop_threshold:
    return 0
  trend_direction = 0
  if uptrend_probability >= gate_threshold or downtrend_probability >= gate_threshold:
    trend_direction = 1 if uptrend_probability >= downtrend_probability else -1
  if trend_direction and base_direction and trend_direction != base_direction:
    return base_direction if countertrend_scale >= 0.5 else 0
  return trend_direction or base_direction


def _calibration_holdout_times(
  common_times: list[str],
  *,
  config: TrendRecognitionComparisonConfig,
) -> tuple[list[str], list[str]]:
  if not common_times:
    return [], []
  if config.calibration_fraction <= 0 or config.calibration_fraction >= 1:
    return [], []
  minimum_rows = max(1, config.minimum_calibration_rows)
  if len(common_times) < 2 * minimum_rows:
    return [], []
  split_index = max(minimum_rows, int(round(len(common_times) * config.calibration_fraction)))
  split_index = min(split_index, len(common_times) - minimum_rows)
  if split_index <= 0 or split_index >= len(common_times):
    return [], []
  return common_times[:split_index], common_times[split_index:]


def _fit_probability_calibrator(
  calibration_times: list[str],
  *,
  labels: dict[str, dict[str, str]],
  trend_by_time: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
) -> tuple[TemperatureScaler | None, dict[str, Any]]:
  method = config.probability_calibration_method.strip().lower()
  if not method or method == "none":
    return None, {"method": "none", "rows": 0, "status": "disabled"}
  if method != "temperature":
    return None, {"method": method, "rows": 0, "status": "unsupported"}

  probability_rows = _trend_probability_rows(
    calibration_times,
    labels=labels,
    trend_by_time=trend_by_time,
    config=config,
  )
  if not probability_rows:
    return None, {"method": method, "rows": 0, "status": "skipped_no_rows"}

  targets = [target for target, _probabilities in probability_rows]
  if len(set(targets)) < 2:
    return None, {"method": method, "rows": len(probability_rows), "status": "skipped_single_class"}

  calibrator = TemperatureScaler()
  try:
    calibrator.fit(
      [list(probabilities) for _target, probabilities in probability_rows],
      targets,
      classes=(-1, 0, 1),
    )
  except ModelDependencyError as exc:
    return None, {"method": method, "rows": len(probability_rows), "status": "skipped_dependency", "reason": str(exc)}

  return calibrator, {
    "method": method,
    "rows": len(probability_rows),
    "classes": list(calibrator.classes),
    "temperature": calibrator.temperature,
    "status": "fit",
  }


def _calibrated_holdout_profile(
  calibration_gate_sweep: list[dict[str, Any]],
  holdout_times: list[str],
  *,
  labels: dict[str, dict[str, str]],
  baseline_by_time: dict[str, dict[str, str]],
  trend_by_time: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  probabilities_by_time: dict[str, tuple[float, float, float]],
  baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
  selected_profile = max(calibration_gate_sweep, key=lambda row: row["optimization_score"], default=None)
  if selected_profile is None or not holdout_times:
    return None
  holdout_metrics = _profile_metrics(
    holdout_times,
    labels=labels,
    baseline_by_time=baseline_by_time,
    trend_by_time=trend_by_time,
    config=config,
    gate_threshold=float(selected_profile["trend_state_gate_threshold"]),
    chop_threshold=float(selected_profile["trend_state_chop_threshold"]),
    countertrend_scale=float(selected_profile["trend_state_countertrend_scale"]),
    probabilities_by_time=probabilities_by_time,
  )
  holdout_score = _optimization_score(holdout_metrics, config=config, baseline_metrics=baseline_metrics)
  return {
    "trend_state_gate_threshold": selected_profile["trend_state_gate_threshold"],
    "trend_state_chop_threshold": selected_profile["trend_state_chop_threshold"],
    "trend_state_countertrend_scale": selected_profile["trend_state_countertrend_scale"],
    "calibration_metrics": selected_profile["metrics"],
    "calibration_score": selected_profile["optimization_score"],
    "holdout_metrics": holdout_metrics,
    "holdout_score": holdout_score,
    "recommended_sizing": {
      "trend_state_filter_enabled": True,
      "trend_state_gate_threshold": selected_profile["trend_state_gate_threshold"],
      "trend_state_chop_threshold": selected_profile["trend_state_chop_threshold"],
      "trend_state_countertrend_scale": selected_profile["trend_state_countertrend_scale"],
    },
  }


def _probabilities_by_time(
  times: list[str],
  *,
  trend_by_time: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  probability_calibrator: TemperatureScaler | None = None,
) -> dict[str, tuple[float, float, float]]:
  output = {
    time: _trend_probabilities(trend_by_time[time], config)
    for time in times
    if time in trend_by_time
  }
  if probability_calibrator is None or not output:
    return output
  ordered_times = list(output)
  calibrated = probability_calibrator.predict_proba([list(output[time]) for time in ordered_times])
  return {
    time: (float(probabilities[0]), float(probabilities[1]), float(probabilities[2]))
    for time, probabilities in zip(ordered_times, calibrated)
  }


def _trend_probability_rows(
  times: list[str],
  *,
  labels: dict[str, dict[str, str]],
  trend_by_time: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
) -> list[tuple[int, tuple[float, float, float]]]:
  rows: list[tuple[int, tuple[float, float, float]]] = []
  for time in times:
    label_row = labels.get(time, {})
    target_value = label_row.get(config.target_column, "")
    if target_value == "" or time not in trend_by_time:
      continue
    rows.append((int(float(target_value)), _trend_probabilities(trend_by_time[time], config)))
  return rows


def _trend_probabilities(row: dict[str, str], config: TrendRecognitionComparisonConfig) -> tuple[float, float, float]:
  return (
    _float(row, config.downtrend_probability_column),
    _float(row, config.chop_probability_column),
    _float(row, config.uptrend_probability_column),
  )


def _probability_calibration_metrics(
  times: list[str],
  *,
  labels: dict[str, dict[str, str]],
  trend_by_time: dict[str, dict[str, str]],
  config: TrendRecognitionComparisonConfig,
  probability_calibrator: TemperatureScaler | None = None,
) -> dict[str, Any]:
  probability_rows = _trend_probability_rows(
    times,
    labels=labels,
    trend_by_time=trend_by_time,
    config=config,
  )
  if probability_calibrator is not None and probability_rows:
    calibrated = probability_calibrator.predict_proba([list(probabilities) for _target, probabilities in probability_rows])
    probability_rows = [
      (target, (float(probabilities[0]), float(probabilities[1]), float(probabilities[2])))
      for (target, _raw_probabilities), probabilities in zip(probability_rows, calibrated)
    ]
  if not probability_rows:
    return {"rows": 0, "accuracy": 0.0, "brier_score": 0.0, "negative_log_likelihood": 0.0, "ece": 0.0, "bins": []}

  labels_order = (-1, 0, 1)
  class_to_index = {label: index for index, label in enumerate(labels_order)}
  correct_count = 0
  brier_total = 0.0
  nll_total = 0.0
  confidence_rows: list[tuple[float, bool]] = []
  for target, probabilities in probability_rows:
    probability_values = [max(0.0, min(1.0, float(probability))) for probability in probabilities]
    predicted_index = max(range(len(probability_values)), key=lambda index: probability_values[index])
    predicted_label = labels_order[predicted_index]
    is_correct = predicted_label == target
    correct_count += int(is_correct)
    confidence_rows.append((probability_values[predicted_index], is_correct))
    target_index = class_to_index.get(target)
    if target_index is None:
      continue
    for index, probability in enumerate(probability_values):
      expected = 1.0 if index == target_index else 0.0
      brier_total += (probability - expected) ** 2
    nll_total -= math.log(max(probability_values[target_index], 1e-9))

  bins = _confidence_bins(confidence_rows)
  ece = sum(
    (bin_row["rows"] / len(confidence_rows)) * abs(bin_row["accuracy"] - bin_row["mean_confidence"])
    for bin_row in bins
  )
  return {
    "rows": len(probability_rows),
    "accuracy": correct_count / len(probability_rows),
    "brier_score": brier_total / len(probability_rows),
    "negative_log_likelihood": nll_total / len(probability_rows),
    "ece": ece,
    "bins": bins,
  }


def _confidence_bins(confidence_rows: list[tuple[float, bool]], *, bin_count: int = 10) -> list[dict[str, Any]]:
  bins: list[dict[str, Any]] = []
  for bin_index in range(bin_count):
    lower = bin_index / bin_count
    upper = (bin_index + 1) / bin_count
    if bin_index == bin_count - 1:
      rows = [(confidence, correct) for confidence, correct in confidence_rows if lower <= confidence <= upper]
    else:
      rows = [(confidence, correct) for confidence, correct in confidence_rows if lower <= confidence < upper]
    if not rows:
      continue
    bins.append(
      {
        "lower": lower,
        "upper": upper,
        "rows": len(rows),
        "mean_confidence": sum(confidence for confidence, _correct in rows) / len(rows),
        "accuracy": sum(1 for _confidence, correct in rows if correct) / len(rows),
      }
    )
  return bins


def _trend_metrics(
  pairs: list[tuple[int, int]],
  *,
  transition_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
  metrics = _classification_metrics(pairs)
  transition_metrics = _classification_metrics(transition_pairs)
  metrics["transition"] = transition_metrics
  return metrics


def _classification_metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
  if not pairs:
    return {
      "rows": 0,
      "accuracy": 0.0,
      "macro_recall": 0.0,
      "trend_direction_accuracy_ex_chop": 0.0,
      "recall": {},
      "target_counts": {},
      "prediction_counts": {},
      "confusion": {},
    }
  labels = sorted(set(target for target, _prediction in pairs) | set(prediction for _target, prediction in pairs))
  recalls = {
    str(label): _class_recall(pairs, label)
    for label in labels
  }
  trend_pairs = [(target, prediction) for target, prediction in pairs if target != 0]
  return {
    "rows": len(pairs),
    "accuracy": sum(1 for target, prediction in pairs if target == prediction) / len(pairs),
    "macro_recall": sum(recalls.values()) / len(recalls) if recalls else 0.0,
    "trend_direction_accuracy_ex_chop": (
      sum(1 for target, prediction in trend_pairs if target == _sign(float(prediction))) / len(trend_pairs)
      if trend_pairs
      else 0.0
    ),
    "recall": recalls,
    "target_counts": {str(label): count for label, count in Counter(target for target, _prediction in pairs).items()},
    "prediction_counts": {str(label): count for label, count in Counter(prediction for _target, prediction in pairs).items()},
    "confusion": _confusion_matrix(pairs, labels),
  }


def _class_recall(pairs: list[tuple[int, int]], label: int) -> float:
  label_pairs = [(target, prediction) for target, prediction in pairs if target == label]
  if not label_pairs:
    return 0.0
  return sum(1 for target, prediction in label_pairs if prediction == target) / len(label_pairs)


def _confusion_matrix(pairs: list[tuple[int, int]], labels: list[int]) -> dict[str, dict[str, int]]:
  matrix = {
    str(target): {str(prediction): 0 for prediction in labels}
    for target in labels
  }
  for target, prediction in pairs:
    matrix[str(target)][str(prediction)] += 1
  return matrix


def _optimization_score(
  metrics: dict[str, Any],
  *,
  config: TrendRecognitionComparisonConfig,
  baseline_metrics: dict[str, Any] | None = None,
) -> float:
  weights = config.optimization_weights or _default_optimization_weights()
  weight_total = sum(abs(float(weight)) for weight in weights.values())
  if weight_total <= 0:
    return 0.0
  weighted_score = sum(
    float(weight) * _metric_value(metrics, metric_path)
    for metric_path, weight in weights.items()
  )
  score = weighted_score / weight_total
  if baseline_metrics:
    score -= _baseline_constraint_penalty(metrics, baseline_metrics=baseline_metrics, config=config)
  return score


def _baseline_constraint_penalty(
  metrics: dict[str, Any],
  *,
  baseline_metrics: dict[str, Any],
  config: TrendRecognitionComparisonConfig,
) -> float:
  penalty = 0.0
  accuracy_floor = _metric_value(baseline_metrics, "accuracy") - max(0.0, config.max_accuracy_drop)
  trend_direction_floor = (
    _metric_value(baseline_metrics, "trend_direction_accuracy_ex_chop")
    - max(0.0, config.max_trend_direction_accuracy_drop)
  )
  if accuracy_floor > 0:
    penalty += max(0.0, accuracy_floor - _metric_value(metrics, "accuracy"))
  if trend_direction_floor > 0:
    penalty += max(0.0, trend_direction_floor - _metric_value(metrics, "trend_direction_accuracy_ex_chop"))
  return max(0.0, config.constraint_penalty_weight) * penalty


def _metric_value(metrics: dict[str, Any], metric_path: str) -> float:
  value: Any = metrics
  for part in metric_path.split("."):
    if not isinstance(value, dict):
      return 0.0
    value = value.get(part)
  if value in (None, ""):
    return 0.0
  return float(value)


def _float(row: dict[str, str], column: str) -> float:
  value = row.get(column, "")
  if value == "":
    return 0.0
  return float(value)


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0
