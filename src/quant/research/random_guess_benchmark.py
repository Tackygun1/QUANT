from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest, simulate_prediction_backtest
from quant.models.base import TrainingArtifactMetadata, utc_run_slug
from quant.risk.sizing import PositionSizingConfig
from quant.training.datasets import SeriesKey, load_series_rows_for_dataset
from quant.validation.metrics import task_metrics


UNIFORM_RANDOM = "uniform_random_class"
SHUFFLED_SIGNAL = "shuffled_model_signal"
BASELINE_MODES = (UNIFORM_RANDOM, SHUFFLED_SIGNAL)


@dataclass(frozen=True, slots=True)
class RandomGuessBenchmarkConfig:
  reference_artifact_dir: Path
  reference_backtest_dir: Path
  output_root: Path
  simulations_per_baseline: int = 500
  seed: int = 20260818
  config_path: Path | None = None


@dataclass(slots=True)
class RandomGuessBenchmarkSummary:
  status: str = "skipped"
  output_dir: Path | None = None
  simulations_completed: int = 0
  representative_artifact_dirs: list[Path] = field(default_factory=list)
  representative_backtest_dirs: list[Path] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GeneratedPrediction:
  hard_prediction: int
  margin: float
  probability_short: float
  probability_flat: float
  probability_long: float
  probability_max: float
  confidence: float


def run_random_guess_benchmark(config: RandomGuessBenchmarkConfig) -> RandomGuessBenchmarkSummary:
  if config.simulations_per_baseline < 1:
    raise ValueError("simulations_per_baseline must be positive")

  reference_metadata = _load_json(config.reference_artifact_dir / "metadata.json")
  reference_predictions = _load_csv(config.reference_artifact_dir / "validation_predictions.csv")
  reference_backtest = _load_json(config.reference_backtest_dir / "summary.json")
  reference_backtest_config = _load_json(config.reference_backtest_dir / "backtest_config.json")
  if reference_metadata.get("task_type") != "classification":
    raise ValueError("Random direction benchmarking requires a classification artifact")

  prediction_column = str(
    reference_metadata.get("extra", {}).get("primary_prediction_column", "prediction")
  )
  dataset_name = str(reference_backtest_config["dataset_name"])
  input_root = Path(reference_backtest_config["input_root"])
  series_key = SeriesKey(
    provider=str(reference_metadata["provider"]),
    symbol=str(reference_metadata["symbol"]),
  )
  dataset_rows = load_series_rows_for_dataset(input_root, series_key, dataset_name=dataset_name)
  dataset_lookup = {row.get("candle_start", ""): row for row in dataset_rows}
  joined_rows: list[dict[str, str]] = []
  matched_predictions: list[dict[str, str]] = []
  for prediction_row in reference_predictions:
    dataset_row = dataset_lookup.get(prediction_row.get("candle_start", ""))
    if dataset_row is None:
      continue
    joined_rows.append(dict(dataset_row))
    matched_predictions.append(prediction_row)
  if not joined_rows:
    return RandomGuessBenchmarkSummary()
  if len(joined_rows) != len(reference_predictions):
    raise ValueError(
      f"Only {len(joined_rows)} of {len(reference_predictions)} reference predictions matched the dataset"
    )

  targets = [int(float(row["target"])) for row in matched_predictions]
  reference_hard_predictions = [int(float(row["prediction"])) for row in matched_predictions]
  reference_margins = [float(row[prediction_column]) for row in matched_predictions]
  sizing = PositionSizingConfig(**reference_backtest_config["sizing"])
  fee_bps = float(reference_backtest_config["fee_bps"])
  slippage_bps = float(reference_backtest_config["slippage_bps"])
  realized_return_column = str(reference_backtest_config["realized_return_column"])

  output_dir = config.output_root / utc_run_slug()
  output_dir.mkdir(parents=True, exist_ok=True)
  summary = RandomGuessBenchmarkSummary(status="completed", output_dir=output_dir)

  representative_summaries: dict[str, dict[str, Any]] = {}
  for mode in BASELINE_MODES:
    generated = generate_random_predictions(
      mode=mode,
      seed=config.seed,
      reference_predictions=matched_predictions,
      reference_hard_predictions=reference_hard_predictions,
      reference_margins=reference_margins,
    )
    artifact_dir = output_dir / "artifacts" / mode / f"seed_{config.seed}"
    _write_random_artifact(
      artifact_dir=artifact_dir,
      mode=mode,
      seed=config.seed,
      reference_artifact_dir=config.reference_artifact_dir,
      reference_metadata=reference_metadata,
      reference_predictions=matched_predictions,
      generated=generated,
      targets=targets,
      dataset_name=dataset_name,
      config_path=config.config_path,
    )
    backtest_summary = run_model_backtest(ModelBacktestConfig(
      artifact_dir=artifact_dir,
      input_root=input_root,
      output_root=output_dir / "representative_backtests",
      dataset_name=dataset_name,
      realized_return_column=realized_return_column,
      fee_bps=fee_bps,
      slippage_bps=slippage_bps,
      sizing=sizing,
      config_path=config.config_path,
    ))
    summary.representative_artifact_dirs.append(artifact_dir)
    if backtest_summary.output_dir is not None:
      summary.representative_backtest_dirs.append(backtest_summary.output_dir)
    representative_summaries[mode] = {
      **asdict(backtest_summary),
      "output_dir": str(backtest_summary.output_dir) if backtest_summary.output_dir else "",
    }

  simulation_rows: list[dict[str, Any]] = []
  for mode in BASELINE_MODES:
    for run_index in range(config.simulations_per_baseline):
      run_seed = config.seed + run_index
      generated = generate_random_predictions(
        mode=mode,
        seed=run_seed,
        reference_predictions=matched_predictions,
        reference_hard_predictions=reference_hard_predictions,
        reference_margins=reference_margins,
      )
      for row, prediction in zip(joined_rows, generated):
        row["prediction"] = str(prediction.margin)
      _, _, aggregate_metrics = simulate_prediction_backtest(
        rows=joined_rows,
        task_type="classification",
        realized_return_column=realized_return_column,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        sizing=sizing,
      )
      predictive_metrics = task_metrics(
        "classification",
        targets,
        [prediction.hard_prediction for prediction in generated],
        prefix="",
      )
      simulation_rows.append({
        "mode": mode,
        "run_index": run_index + 1,
        "seed": run_seed,
        "accuracy": predictive_metrics["accuracy"],
        "macro_recall": predictive_metrics["macro_recall"],
        "net_return": aggregate_metrics["total_net_return"],
        "gross_return": aggregate_metrics["total_gross_return"],
        "transaction_cost": aggregate_metrics["total_transaction_cost"],
        "max_drawdown": aggregate_metrics["max_drawdown"],
        "sharpe_like": aggregate_metrics["sharpe_like"],
        "profit_factor_after_cost": aggregate_metrics["profit_factor_after_cost"],
        "active_hit_rate": aggregate_metrics["active_hit_rate"],
        "active_periods": int(aggregate_metrics["active_periods"]),
        "position_adjustments": int(aggregate_metrics["trade_count"]),
        "entries": int(aggregate_metrics["entry_count"]),
        "total_turnover": aggregate_metrics["total_turnover"],
        "drawdown_halt_periods": int(aggregate_metrics["drawdown_halt_periods"]),
      })
  summary.simulations_completed = len(simulation_rows)
  _write_csv(output_dir / "simulations.csv", simulation_rows)

  reference_metrics = {
    "accuracy": float(reference_metadata["metrics"]["overall_oof_accuracy"]),
    "macro_recall": float(reference_metadata["metrics"]["overall_oof_macro_recall"]),
    "net_return": float(reference_backtest["total_net_return"]),
    "gross_return": float(reference_backtest["aggregate_metrics"]["total_gross_return"]),
    "transaction_cost": float(reference_backtest["total_transaction_cost"]),
    "max_drawdown": float(reference_backtest["max_drawdown"]),
    "sharpe_like": float(reference_backtest["sharpe_like"]),
    "profit_factor_after_cost": float(reference_backtest["profit_factor_after_cost"]),
    "active_hit_rate": float(reference_backtest["active_hit_rate"]),
    "active_periods": int(reference_backtest["active_periods"]),
    "position_adjustments": int(reference_backtest["trades"]),
    "entries": int(reference_backtest["aggregate_metrics"]["entry_count"]),
    "total_turnover": float(reference_backtest["total_turnover"]),
  }
  mode_summaries = {
    mode: _summarize_mode(
      [row for row in simulation_rows if row["mode"] == mode],
      reference_metrics=reference_metrics,
    )
    for mode in BASELINE_MODES
  }
  payload = {
    "status": "completed",
    "reference_artifact_dir": str(config.reference_artifact_dir),
    "reference_backtest_dir": str(config.reference_backtest_dir),
    "config_path": str(config.config_path) if config.config_path else "",
    "seed": config.seed,
    "simulations_per_baseline": config.simulations_per_baseline,
    "oof_rows": len(joined_rows),
    "fee_bps": fee_bps,
    "slippage_bps": slippage_bps,
    "reference_model": reference_metrics,
    "baselines": mode_summaries,
    "representative_runs": representative_summaries,
    "methodology": {
      UNIFORM_RANDOM: "Independent uniform draws from short, flat, and long with one-hot confidence.",
      SHUFFLED_SIGNAL: "Random permutation of the reference model's full hard-prediction, probability, confidence, and margin tuples across the same OOF timestamps.",
    },
  }
  (output_dir / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  (output_dir / "REPORT.md").write_text(_build_report(payload), encoding="utf-8")
  return summary


def generate_random_predictions(
  *,
  mode: str,
  seed: int,
  reference_predictions: list[dict[str, str]],
  reference_hard_predictions: list[int],
  reference_margins: list[float],
) -> list[GeneratedPrediction]:
  rng = random.Random(seed)
  if mode == UNIFORM_RANDOM:
    generated = []
    for _ in reference_predictions:
      hard_prediction = rng.choice((-1, 0, 1))
      generated.append(GeneratedPrediction(
        hard_prediction=hard_prediction,
        margin=float(hard_prediction),
        probability_short=float(hard_prediction == -1),
        probability_flat=float(hard_prediction == 0),
        probability_long=float(hard_prediction == 1),
        probability_max=1.0,
        confidence=1.0,
      ))
    return generated
  if mode != SHUFFLED_SIGNAL:
    raise ValueError(f"Unknown random baseline mode: {mode}")

  indices = list(range(len(reference_predictions)))
  rng.shuffle(indices)
  generated = []
  for source_index in indices:
    row = reference_predictions[source_index]
    generated.append(GeneratedPrediction(
      hard_prediction=reference_hard_predictions[source_index],
      margin=reference_margins[source_index],
      probability_short=float(row.get("prediction_probability_short", 0.0)),
      probability_flat=float(row.get("prediction_probability_flat", 0.0)),
      probability_long=float(row.get("prediction_probability_long", 0.0)),
      probability_max=float(row.get("prediction_probability_max", 0.0)),
      confidence=float(row.get("prediction_confidence", 0.0)),
    ))
  return generated


def _write_random_artifact(
  *,
  artifact_dir: Path,
  mode: str,
  seed: int,
  reference_artifact_dir: Path,
  reference_metadata: dict[str, Any],
  reference_predictions: list[dict[str, str]],
  generated: list[GeneratedPrediction],
  targets: list[int],
  dataset_name: str,
  config_path: Path | None,
) -> None:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  prediction_rows = []
  for reference_row, prediction, target in zip(reference_predictions, generated, targets):
    prediction_rows.append({
      "provider": reference_row.get("provider", ""),
      "symbol": reference_row.get("symbol", ""),
      "candle_start": reference_row.get("candle_start", ""),
      "target": target,
      "prediction": prediction.hard_prediction,
      "prediction_probability_max": prediction.probability_max,
      "prediction_confidence": prediction.confidence,
      "prediction_probability_short": prediction.probability_short,
      "prediction_probability_flat": prediction.probability_flat,
      "prediction_probability_long": prediction.probability_long,
      "prediction_margin": prediction.margin,
      "fold_index": reference_row.get("fold_index", ""),
    })
  _write_csv(artifact_dir / "validation_predictions.csv", prediction_rows)
  metrics = task_metrics(
    "classification",
    targets,
    [prediction.hard_prediction for prediction in generated],
    prefix="overall_oof_",
  )
  TrainingArtifactMetadata(
    model_name="random_direction_baseline" if mode == UNIFORM_RANDOM else "shuffled_hybrid_signal_baseline",
    task_type="classification",
    target_column=str(reference_metadata["target_column"]),
    provider=str(reference_metadata["provider"]),
    symbol=str(reference_metadata["symbol"]),
    feature_columns=(),
    train_rows=0,
    validation_rows=len(prediction_rows),
    metrics=metrics,
    extra={
      "baseline_mode": mode,
      "config_path": str(config_path) if config_path else "",
      "dataset_name": dataset_name,
      "evaluation_type": "random_oof_timestamp_baseline",
      "primary_prediction_column": "prediction_margin",
      "hard_prediction_column": "prediction",
      "confidence_column": "prediction_confidence",
      "probability_columns": [
        "prediction_probability_short",
        "prediction_probability_flat",
        "prediction_probability_long",
      ],
      "random_seed": seed,
      "reference_artifact_dir": str(reference_artifact_dir),
    },
  ).write_json(artifact_dir / "metadata.json")


def _summarize_mode(
  rows: list[dict[str, Any]],
  *,
  reference_metrics: dict[str, float | int],
) -> dict[str, Any]:
  metric_names = (
    "accuracy",
    "macro_recall",
    "net_return",
    "gross_return",
    "transaction_cost",
    "max_drawdown",
    "sharpe_like",
    "profit_factor_after_cost",
    "active_hit_rate",
    "active_periods",
    "position_adjustments",
    "entries",
    "total_turnover",
    "drawdown_halt_periods",
  )
  distributions = {
    metric: _distribution([float(row[metric]) for row in rows])
    for metric in metric_names
  }
  count = len(rows)
  return {
    "runs": count,
    "distributions": distributions,
    "comparison_to_reference": {
      "probability_random_accuracy_at_least_reference": (
        sum(float(row["accuracy"]) >= float(reference_metrics["accuracy"]) for row in rows) / count
      ),
      "probability_random_net_return_at_least_reference": (
        sum(float(row["net_return"]) >= float(reference_metrics["net_return"]) for row in rows) / count
      ),
      "one_sided_empirical_p_accuracy": (
        1 + sum(float(row["accuracy"]) >= float(reference_metrics["accuracy"]) for row in rows)
      ) / (count + 1),
      "one_sided_empirical_p_net_return": (
        1 + sum(float(row["net_return"]) >= float(reference_metrics["net_return"]) for row in rows)
      ) / (count + 1),
      "profitable_run_fraction": sum(float(row["net_return"]) > 0 for row in rows) / count,
      "drawdown_limit_breach_fraction": sum(float(row["max_drawdown"]) >= 0.01 for row in rows) / count,
    },
  }


def _distribution(values: list[float]) -> dict[str, float]:
  ordered = sorted(values)
  return {
    "min": ordered[0],
    "p025": _quantile(ordered, 0.025),
    "p05": _quantile(ordered, 0.05),
    "p25": _quantile(ordered, 0.25),
    "median": _quantile(ordered, 0.5),
    "mean": statistics.fmean(ordered),
    "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    "p75": _quantile(ordered, 0.75),
    "p95": _quantile(ordered, 0.95),
    "p975": _quantile(ordered, 0.975),
    "max": ordered[-1],
  }


def _quantile(ordered: list[float], probability: float) -> float:
  position = (len(ordered) - 1) * probability
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _build_report(payload: dict[str, Any]) -> str:
  reference = payload["reference_model"]
  uniform = payload["baselines"][UNIFORM_RANDOM]
  shuffled = payload["baselines"][SHUFFLED_SIGNAL]

  def metric(mode: dict[str, Any], name: str, statistic: str = "median") -> float:
    return float(mode["distributions"][name][statistic])

  return "\n".join([
    "# Random-Guess Baseline Comparison",
    "",
    "## Method",
    "",
    f"The comparison uses {payload['oof_rows']:,} identical OOF timestamps, the same market rows, "
    f"{payload['fee_bps'] + payload['slippage_bps']:.1f} bps total execution cost, and the same 1% drawdown limit.",
    "",
    f"- Uniform baseline: {payload['methodology'][UNIFORM_RANDOM]}",
    f"- Shuffled-signal baseline: {payload['methodology'][SHUFFLED_SIGNAL]}",
    f"- Monte Carlo runs per baseline: {payload['simulations_per_baseline']}",
    "",
    "## Comparison",
    "",
    "| Metric | Hybrid OOF model | Uniform random median | Shuffled-signal median |",
    "|---|---:|---:|---:|",
    f"| Accuracy | {100 * reference['accuracy']:.2f}% | {100 * metric(uniform, 'accuracy'):.2f}% | {100 * metric(shuffled, 'accuracy'):.2f}% |",
    f"| Gross return | {100 * reference['gross_return']:+.3f}% | {100 * metric(uniform, 'gross_return'):+.3f}% | {100 * metric(shuffled, 'gross_return'):+.3f}% |",
    f"| Net return | {100 * reference['net_return']:+.3f}% | {100 * metric(uniform, 'net_return'):+.3f}% | {100 * metric(shuffled, 'net_return'):+.3f}% |",
    f"| Maximum drawdown | {100 * reference['max_drawdown']:.3f}% | {100 * metric(uniform, 'max_drawdown'):.3f}% | {100 * metric(shuffled, 'max_drawdown'):.3f}% |",
    f"| Position adjustments | {reference['position_adjustments']} | {metric(uniform, 'position_adjustments'):.0f} | {metric(shuffled, 'position_adjustments'):.0f} |",
    f"| Transaction cost | {100 * reference['transaction_cost']:.3f}% | {100 * metric(uniform, 'transaction_cost'):.3f}% | {100 * metric(shuffled, 'transaction_cost'):.3f}% |",
    "",
    "## Chance comparison",
    "",
    f"- Uniform random one-sided empirical p-value for net return: {uniform['comparison_to_reference']['one_sided_empirical_p_net_return']:.6f}",
    f"- Shuffled-signal one-sided empirical p-value for net return: {shuffled['comparison_to_reference']['one_sided_empirical_p_net_return']:.6f}",
    f"- Uniform random profitable-run fraction: {100 * uniform['comparison_to_reference']['profitable_run_fraction']:.1f}%",
    f"- Shuffled-signal profitable-run fraction: {100 * shuffled['comparison_to_reference']['profitable_run_fraction']:.1f}%",
    f"- Uniform random 1% drawdown-limit breach fraction: {100 * uniform['comparison_to_reference']['drawdown_limit_breach_fraction']:.1f}%",
    f"- Shuffled-signal 1% drawdown-limit breach fraction: {100 * shuffled['comparison_to_reference']['drawdown_limit_breach_fraction']:.1f}%",
    "",
    "The shuffled-signal result is the more meaningful financial null because it preserves the hybrid model's pre-execution signal-strength distribution while destroying timestamp alignment.",
    "",
  ])


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0]) if rows else []
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if fieldnames:
      writer.writeheader()
      writer.writerows(rows)
