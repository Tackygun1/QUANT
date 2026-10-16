from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest, simulate_prediction_backtest
from quant.models.base import utc_run_slug
from quant.risk.sizing import PositionSizingConfig
from quant.training.datasets import SeriesKey, load_series_rows_for_dataset


@dataclass(frozen=True, slots=True)
class ProfitPerTradeRequirements:
  minimum_entries: int = 100
  minimum_traded_folds: int = 12
  minimum_positive_folds: int = 9
  minimum_median_fold_return: float = 0.0
  maximum_best_fold_profit_share: float = 0.4
  maximum_drawdown: float = 0.01
  stress_total_cost_bps: float = 3.0
  minimum_stress_net_return: float = 0.0
  minimum_net_edge_per_turnover_bps: float = 2.5


@dataclass(frozen=True, slots=True)
class ProfitPerTradeSweepConfig:
  reference_artifact_dir: Path
  reference_backtest_dir: Path
  output_root: Path
  margin_thresholds: tuple[float, ...] = (0.12, 0.16, 0.2, 0.24)
  minimum_hold_bars: tuple[int, ...] = (2, 6, 12)
  minimum_rebalance_deltas: tuple[float, ...] = (0.025, 0.05, 0.1)
  fixed_target_positions: tuple[float, ...] = (0.0, 0.25)
  forecast_gate_profiles: tuple[tuple[bool, float, float], ...] = (
    (False, 3.0, 5.0),
    (True, 3.0, 5.0),
    (True, 5.0, 8.0),
    (True, 5.0, 10.0),
  )
  requirements: ProfitPerTradeRequirements = ProfitPerTradeRequirements()
  stress_fallback_candidates: int = 20
  config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SweepVariant:
  slug: str
  sizing: PositionSizingConfig


def run_profit_per_trade_sweep(config: ProfitPerTradeSweepConfig) -> dict[str, Any]:
  reference_metadata = _load_json(config.reference_artifact_dir / "metadata.json")
  reference_predictions = _load_csv(config.reference_artifact_dir / "validation_predictions.csv")
  reference_backtest_config = _load_json(config.reference_backtest_dir / "backtest_config.json")
  reference_backtest_summary = _load_json(config.reference_backtest_dir / "summary.json")
  if reference_metadata.get("task_type") != "classification":
    raise ValueError("Profit-per-trade tuning requires a classification artifact")

  input_root = Path(reference_backtest_config["input_root"])
  dataset_name = str(reference_backtest_config["dataset_name"])
  realized_return_column = str(reference_backtest_config["realized_return_column"])
  fee_bps = float(reference_backtest_config["fee_bps"])
  slippage_bps = float(reference_backtest_config["slippage_bps"])
  base_sizing = PositionSizingConfig(**reference_backtest_config["sizing"])
  prediction_column = str(
    reference_metadata.get("extra", {}).get("primary_prediction_column", "prediction")
  )
  series_key = SeriesKey(
    provider=str(reference_metadata["provider"]),
    symbol=str(reference_metadata["symbol"]),
  )
  dataset_rows = load_series_rows_for_dataset(input_root, series_key, dataset_name=dataset_name)
  dataset_lookup = {row.get("candle_start", ""): row for row in dataset_rows}
  joined_rows: list[dict[str, str]] = []
  fold_lookup: dict[str, int] = {}
  for prediction_row in reference_predictions:
    candle_start = prediction_row.get("candle_start", "")
    dataset_row = dataset_lookup.get(candle_start)
    if dataset_row is None:
      continue
    row = dict(dataset_row)
    row["prediction"] = prediction_row[prediction_column]
    joined_rows.append(row)
    fold_lookup[candle_start] = int(prediction_row.get("fold_index", 0))
  if len(joined_rows) != len(reference_predictions):
    raise ValueError(
      f"Only {len(joined_rows)} of {len(reference_predictions)} reference predictions matched the dataset"
    )
  if not fold_lookup or min(fold_lookup.values()) < 1:
    raise ValueError("Every reference prediction must include a positive fold_index")

  output_dir = config.output_root / utc_run_slug()
  output_dir.mkdir(parents=True, exist_ok=True)
  variants = build_profit_per_trade_variants(config=config, base_sizing=base_sizing)
  evaluations: list[tuple[SweepVariant, dict[str, Any]]] = []
  for variant in variants:
    period_rows, _, aggregate_metrics = simulate_prediction_backtest(
      rows=joined_rows,
      task_type="classification",
      realized_return_column=realized_return_column,
      fee_bps=fee_bps,
      slippage_bps=slippage_bps,
      sizing=variant.sizing,
    )
    metrics = {
      "variant_slug": variant.slug,
      **_candidate_parameters(variant.sizing),
      **_candidate_metrics(
        aggregate_metrics=aggregate_metrics,
        period_rows=period_rows,
        fold_lookup=fold_lookup,
      ),
      "stress_net_return": None,
      "stress_max_drawdown": None,
      "stress_position_adjustments": None,
    }
    metrics["pre_stress_failures"] = _requirement_failures(
      metrics,
      requirements=config.requirements,
      include_stress=False,
    )
    evaluations.append((variant, metrics))

  preliminary = [item for item in evaluations if not item[1]["pre_stress_failures"]]
  stress_variants = list(preliminary)
  if len(stress_variants) < config.stress_fallback_candidates:
    fallback_pool = sorted(
      (
        item for item in evaluations
        if int(item[1]["entries"]) >= config.requirements.minimum_entries
        and int(item[1]["traded_folds"]) >= config.requirements.minimum_traded_folds
        and item not in stress_variants
      ),
      key=lambda item: _base_candidate_sort_key(item[1]),
      reverse=True,
    )
    stress_variants.extend(
      fallback_pool[: config.stress_fallback_candidates - len(stress_variants)]
    )

  stress_fee_bps = max(0.0, config.requirements.stress_total_cost_bps - slippage_bps)
  for variant, metrics in stress_variants:
    _, _, stress_metrics = simulate_prediction_backtest(
      rows=joined_rows,
      task_type="classification",
      realized_return_column=realized_return_column,
      fee_bps=stress_fee_bps,
      slippage_bps=slippage_bps,
      sizing=variant.sizing,
    )
    metrics["stress_net_return"] = float(stress_metrics["total_net_return"])
    metrics["stress_max_drawdown"] = float(stress_metrics["max_drawdown"])
    metrics["stress_position_adjustments"] = int(stress_metrics["trade_count"])

  for _, metrics in evaluations:
    failures = _requirement_failures(
      metrics,
      requirements=config.requirements,
      include_stress=True,
    )
    metrics["requirement_failures"] = failures
    metrics["requirements_passed"] = _requirements_passed_count(failures)
    metrics["feasible"] = not failures

  evaluations.sort(key=lambda item: _final_candidate_sort_key(item[1]), reverse=True)
  feasible = [item for item in evaluations if bool(item[1]["feasible"])]
  activity_safe = [
    item for item in evaluations
    if int(item[1]["entries"]) >= config.requirements.minimum_entries
    and int(item[1]["traded_folds"]) >= config.requirements.minimum_traded_folds
  ]
  stress_tested_activity_safe = [
    item for item in activity_safe
    if item[1]["stress_net_return"] is not None
  ]
  recommended = (
    feasible[0]
    if feasible
    else (
      stress_tested_activity_safe[0]
      if stress_tested_activity_safe
      else (activity_safe[0] if activity_safe else evaluations[0])
    )
  )
  recommended_variant, recommended_metrics = recommended
  selection_status = "all_requirements_passed" if feasible else "no_fully_feasible_candidate"

  recommended_backtest = run_model_backtest(ModelBacktestConfig(
    artifact_dir=config.reference_artifact_dir,
    input_root=input_root,
    output_root=output_dir / "recommended_backtest",
    dataset_name=dataset_name,
    realized_return_column=realized_return_column,
    fee_bps=fee_bps,
    slippage_bps=slippage_bps,
    sizing=recommended_variant.sizing,
    config_path=config.config_path,
  ))
  recommended_stress_backtest = run_model_backtest(ModelBacktestConfig(
    artifact_dir=config.reference_artifact_dir,
    input_root=input_root,
    output_root=output_dir / "recommended_stress_backtest",
    dataset_name=dataset_name,
    realized_return_column=realized_return_column,
    fee_bps=stress_fee_bps,
    slippage_bps=slippage_bps,
    sizing=recommended_variant.sizing,
    config_path=config.config_path,
  ))

  candidate_rows = [_csv_candidate_row(metrics) for _, metrics in evaluations]
  _write_csv(output_dir / "candidates.csv", candidate_rows)
  selected_config_payload = {
    **reference_backtest_config,
    "artifact_dir": str(config.reference_artifact_dir),
    "output_root": str(output_dir / "deployment_validation"),
    "sizing": asdict(recommended_variant.sizing),
    "selection": {
      "status": selection_status,
      "variant_slug": recommended_variant.slug,
      "requirements_passed": bool(recommended_metrics["feasible"]),
      "requirement_failures": recommended_metrics["requirement_failures"],
      "source_sweep": str(output_dir),
    },
  }
  config_filename = (
    "selected_execution_config.json"
    if feasible
    else "best_activity_safe_config_NOT_APPROVED.json"
  )
  (output_dir / config_filename).write_text(
    json.dumps(selected_config_payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )

  reference_metrics = _reference_metrics(
    reference_backtest_summary=reference_backtest_summary,
    fold_lookup=fold_lookup,
    reference_backtest_dir=config.reference_backtest_dir,
  )
  _, _, reference_stress_metrics = simulate_prediction_backtest(
    rows=joined_rows,
    task_type="classification",
    realized_return_column=realized_return_column,
    fee_bps=stress_fee_bps,
    slippage_bps=slippage_bps,
    sizing=base_sizing,
  )
  reference_metrics["stress_net_return"] = float(reference_stress_metrics["total_net_return"])
  payload = {
    "status": "completed",
    "selection_status": selection_status,
    "config_path": str(config.config_path) if config.config_path else "",
    "reference_artifact_dir": str(config.reference_artifact_dir),
    "reference_backtest_dir": str(config.reference_backtest_dir),
    "output_dir": str(output_dir),
    "candidate_count": len(evaluations),
    "stress_tested_candidates": len(stress_variants),
    "feasible_candidate_count": len(feasible),
    "activity_safe_candidate_count": len(activity_safe),
    "requirements": asdict(config.requirements),
    "grid": {
      "margin_thresholds": config.margin_thresholds,
      "minimum_hold_bars": config.minimum_hold_bars,
      "minimum_rebalance_deltas": config.minimum_rebalance_deltas,
      "fixed_target_positions": config.fixed_target_positions,
      "forecast_gate_profiles": config.forecast_gate_profiles,
    },
    "reference_metrics": reference_metrics,
    "recommended_candidate": {
      **recommended_metrics,
      "sizing": asdict(recommended_variant.sizing),
      "backtest_dir": str(recommended_backtest.output_dir),
      "stress_backtest_dir": str(recommended_stress_backtest.output_dir),
      "config_file": str(output_dir / config_filename),
    },
    "top_candidates": [
      {**metrics, "sizing": asdict(variant.sizing)}
      for variant, metrics in evaluations[:10]
    ],
  }
  (output_dir / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  (output_dir / "REPORT.md").write_text(_build_report(payload), encoding="utf-8")
  return payload


def build_profit_per_trade_variants(
  *,
  config: ProfitPerTradeSweepConfig,
  base_sizing: PositionSizingConfig,
) -> list[SweepVariant]:
  variants: list[SweepVariant] = []
  seen: set[str] = set()
  for margin_threshold in config.margin_thresholds:
    for min_hold_bars in config.minimum_hold_bars:
      for min_rebalance in config.minimum_rebalance_deltas:
        for fixed_target in config.fixed_target_positions:
          for forecast_enabled, fast_threshold, slow_threshold in config.forecast_gate_profiles:
            sizing = replace(
              base_sizing,
              classification_margin_threshold=margin_threshold,
              min_hold_bars=min_hold_bars,
              min_rebalance_abs_delta=min_rebalance,
              classification_fixed_target_position=fixed_target,
              forecast_gate_enabled=forecast_enabled,
              forecast_gate_fast_threshold_bps=fast_threshold,
              forecast_gate_slow_threshold_bps=slow_threshold,
              forecast_gate_require_alignment=True,
            )
            gate_slug = (
              f"gate{fast_threshold:g}x{slow_threshold:g}"
              if forecast_enabled
              else "gateoff"
            )
            slug = (
              f"margin{margin_threshold:g}_hold{min_hold_bars}_"
              f"rebal{min_rebalance:g}_fixed{fixed_target:g}_{gate_slug}"
            )
            if slug in seen:
              continue
            seen.add(slug)
            variants.append(SweepVariant(slug=slug, sizing=sizing))
  return variants


def _candidate_metrics(
  *,
  aggregate_metrics: dict[str, float],
  period_rows: list[dict[str, Any]],
  fold_lookup: dict[str, int],
) -> dict[str, Any]:
  fold_growth: dict[int, float] = {fold_index: 1.0 for fold_index in sorted(set(fold_lookup.values()))}
  fold_entries: dict[int, int] = {fold_index: 0 for fold_index in fold_growth}
  previous_position = 0.0
  for row in period_rows:
    fold_index = fold_lookup[str(row["candle_start"])]
    fold_growth[fold_index] *= 1.0 + float(row["net_return"])
    position = float(row["target_position"])
    if previous_position == 0.0 and position != 0.0:
      fold_entries[fold_index] += 1
    previous_position = position
  fold_returns = [fold_growth[index] - 1.0 for index in sorted(fold_growth)]
  positive_fold_returns = [value for value in fold_returns if value > 0.0]
  positive_profit = sum(positive_fold_returns)
  best_fold_profit_share = (
    max(positive_fold_returns) / positive_profit
    if positive_profit > 0.0
    else 1.0
  )
  net_return = float(aggregate_metrics["total_net_return"])
  turnover = float(aggregate_metrics["total_turnover"])
  entries = int(aggregate_metrics["entry_count"])
  adjustments = int(aggregate_metrics["trade_count"])
  return {
    "net_return": net_return,
    "gross_return": float(aggregate_metrics["total_gross_return"]),
    "transaction_cost": float(aggregate_metrics["total_transaction_cost"]),
    "max_drawdown": float(aggregate_metrics["max_drawdown"]),
    "sharpe_like": float(aggregate_metrics["sharpe_like"]),
    "profit_factor_after_cost": float(aggregate_metrics["profit_factor_after_cost"]),
    "total_turnover": turnover,
    "entries": entries,
    "position_adjustments": adjustments,
    "active_periods": int(aggregate_metrics["active_periods"]),
    "traded_folds": sum(count > 0 for count in fold_entries.values()),
    "positive_folds": len(positive_fold_returns),
    "median_fold_return": statistics.median(fold_returns),
    "best_fold_profit_share": best_fold_profit_share,
    "net_edge_per_turnover_bps": (net_return / turnover * 10_000) if turnover > 0.0 else 0.0,
    "net_profit_per_entry_bps": (net_return / entries * 10_000) if entries > 0 else 0.0,
    "net_profit_per_adjustment_bps": (net_return / adjustments * 10_000) if adjustments > 0 else 0.0,
    "fold_returns": fold_returns,
    "fold_entries": [fold_entries[index] for index in sorted(fold_entries)],
  }


def _requirement_failures(
  metrics: dict[str, Any],
  *,
  requirements: ProfitPerTradeRequirements,
  include_stress: bool,
) -> list[str]:
  failures = []
  if int(metrics["entries"]) < requirements.minimum_entries:
    failures.append("minimum_entries")
  if int(metrics["traded_folds"]) < requirements.minimum_traded_folds:
    failures.append("minimum_traded_folds")
  if int(metrics["positive_folds"]) < requirements.minimum_positive_folds:
    failures.append("minimum_positive_folds")
  if float(metrics["median_fold_return"]) <= requirements.minimum_median_fold_return:
    failures.append("positive_median_fold_return")
  if float(metrics["best_fold_profit_share"]) > requirements.maximum_best_fold_profit_share:
    failures.append("maximum_best_fold_profit_share")
  if float(metrics["max_drawdown"]) > requirements.maximum_drawdown:
    failures.append("maximum_drawdown")
  if float(metrics["net_edge_per_turnover_bps"]) < requirements.minimum_net_edge_per_turnover_bps:
    failures.append("minimum_net_edge_per_turnover_bps")
  if include_stress:
    stress_net_return = metrics.get("stress_net_return")
    if stress_net_return is None:
      failures.append("stress_not_evaluated")
    elif float(stress_net_return) <= requirements.minimum_stress_net_return:
      failures.append("positive_stress_net_return")
  return failures


def _requirements_passed_count(failures: list[str]) -> int:
  return 8 - len(failures)


def _base_candidate_sort_key(metrics: dict[str, Any]) -> tuple[float, ...]:
  return (
    -float(len(metrics["pre_stress_failures"])),
    float(metrics["positive_folds"]),
    float(metrics["median_fold_return"]),
    float(metrics["net_edge_per_turnover_bps"]),
    float(metrics["net_return"]),
  )


def _final_candidate_sort_key(metrics: dict[str, Any]) -> tuple[float, ...]:
  stress_net_return = metrics.get("stress_net_return")
  return (
    float(bool(metrics["feasible"])),
    float(metrics["requirements_passed"]),
    float(metrics["positive_folds"]),
    float(metrics["median_fold_return"]),
    float(stress_net_return) if stress_net_return is not None else -1.0,
    float(metrics["net_edge_per_turnover_bps"]),
    float(metrics["net_return"]),
    -float(metrics["total_turnover"]),
  )


def _candidate_parameters(sizing: PositionSizingConfig) -> dict[str, Any]:
  return {
    "margin_threshold": sizing.classification_margin_threshold,
    "minimum_hold_bars": sizing.min_hold_bars,
    "minimum_rebalance_delta": sizing.min_rebalance_abs_delta,
    "fixed_target_position": sizing.classification_fixed_target_position,
    "forecast_gate_enabled": sizing.forecast_gate_enabled,
    "forecast_fast_threshold_bps": sizing.forecast_gate_fast_threshold_bps,
    "forecast_slow_threshold_bps": sizing.forecast_gate_slow_threshold_bps,
  }


def _csv_candidate_row(metrics: dict[str, Any]) -> dict[str, Any]:
  excluded = {"fold_returns", "fold_entries", "pre_stress_failures", "requirement_failures"}
  row = {key: value for key, value in metrics.items() if key not in excluded}
  row["pre_stress_failures"] = "|".join(metrics["pre_stress_failures"])
  row["requirement_failures"] = "|".join(metrics["requirement_failures"])
  row["fold_returns"] = "|".join(f"{value:.12g}" for value in metrics["fold_returns"])
  row["fold_entries"] = "|".join(str(value) for value in metrics["fold_entries"])
  return row


def _reference_metrics(
  *,
  reference_backtest_summary: dict[str, Any],
  fold_lookup: dict[str, int],
  reference_backtest_dir: Path,
) -> dict[str, Any]:
  period_rows = _load_csv(reference_backtest_dir / "periods.csv")
  aggregate_metrics = reference_backtest_summary["aggregate_metrics"]
  return _candidate_metrics(
    aggregate_metrics=aggregate_metrics,
    period_rows=period_rows,
    fold_lookup=fold_lookup,
  )


def _build_report(payload: dict[str, Any]) -> str:
  reference = payload["reference_metrics"]
  candidate = payload["recommended_candidate"]
  requirements = payload["requirements"]
  passed = payload["selection_status"] == "all_requirements_passed"
  status_text = (
    "A candidate passed every hard requirement."
    if passed
    else "No candidate passed every hard requirement; the reported candidate is activity-safe but is not approved."
  )
  return "\n".join([
    "# Constrained Profit-per-Trade OOF Sweep",
    "",
    "## Result",
    "",
    status_text,
    "",
    f"- Candidates evaluated: {payload['candidate_count']}",
    f"- Fully feasible candidates: {payload['feasible_candidate_count']}",
    f"- Candidates with at least {requirements['minimum_entries']} entries and {requirements['minimum_traded_folds']} traded folds: {payload['activity_safe_candidate_count']}",
    f"- Recommended variant: `{candidate['variant_slug']}`",
    f"- Failed requirements: {', '.join(candidate['requirement_failures']) if candidate['requirement_failures'] else 'none'}",
    "",
    "## Comparison",
    "",
    "| Metric | Original | Recommended |",
    "|---|---:|---:|",
    f"| Net return at 1.5 bps cost | {100 * reference['net_return']:+.3f}% | {100 * candidate['net_return']:+.3f}% |",
    f"| Net return at 3.0 bps cost | {100 * reference['stress_net_return']:+.3f}% | {100 * float(candidate['stress_net_return'] or 0.0):+.3f}% |",
    f"| Max drawdown | {100 * reference['max_drawdown']:.3f}% | {100 * candidate['max_drawdown']:.3f}% |",
    f"| Entries | {reference['entries']} | {candidate['entries']} |",
    f"| Position adjustments | {reference['position_adjustments']} | {candidate['position_adjustments']} |",
    f"| Traded folds | {reference['traded_folds']} | {candidate['traded_folds']} |",
    f"| Positive folds | {reference['positive_folds']} | {candidate['positive_folds']} |",
    f"| Median fold return | {100 * reference['median_fold_return']:+.4f}% | {100 * candidate['median_fold_return']:+.4f}% |",
    f"| Best-fold share of positive fold profit | {100 * reference['best_fold_profit_share']:.1f}% | {100 * candidate['best_fold_profit_share']:.1f}% |",
    f"| Net edge per turnover | {reference['net_edge_per_turnover_bps']:.3f} bps | {candidate['net_edge_per_turnover_bps']:.3f} bps |",
    f"| Net profit per entry | {reference['net_profit_per_entry_bps']:.3f} bps | {candidate['net_profit_per_entry_bps']:.3f} bps |",
    "",
    "## Anti-starvation rule",
    "",
    f"Candidates are rejected below {requirements['minimum_entries']} entries or below {requirements['minimum_traded_folds']} traded folds. A low-trade candidate cannot win by producing a deceptively high profit-per-trade statistic from a handful of observations.",
    "",
    "This sweep is exploratory because it selects execution parameters on the same OOF stream it scores. Confirm any passing configuration on a later untouched forward period before deployment.",
    "",
  ])


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = list(rows[0]) if rows else []
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if fieldnames:
      writer.writeheader()
      writer.writerows(rows)
