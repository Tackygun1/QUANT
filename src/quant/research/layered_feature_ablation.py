from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest
from quant.models.base import utc_run_slug
from quant.models.transformers.hybrid_forecaster import (
  HybridTransformerConfig,
  build_prediction_columns,
  build_target_columns,
)
from quant.risk.sizing import PositionSizingConfig
from quant.training.stacked_dataset import (
  PredictionSource,
  StackedDatasetConfig,
  build_stacked_datasets,
)
from quant.training.train_meta import MetaTrainingConfig, train_meta_models
from quant.validation.hybrid_walk_forward import (
  HybridTransformerWalkForwardConfig,
  evaluate_hybrid_transformer_walk_forward,
)


DEFAULT_PRUNED_STRUCTURE_COLUMNS = (
  "structure_close_vs_session_vwap_pct",
  "structure_range_position_24_bar",
  "structure_distance_to_high_24_bar",
  "structure_distance_to_low_24_bar",
  "structure_breakout_up_24_bar",
  "structure_breakout_down_24_bar",
  "structure_range_position_96_bar",
  "structure_distance_to_high_96_bar",
  "structure_distance_to_low_96_bar",
  "structure_breakout_up_96_bar",
  "structure_breakout_down_96_bar",
)

DEFAULT_META_PARAMS = {
  "learning_rate": 0.035317416307905206,
  "n_estimators": 150,
  "num_leaves": 15,
  "min_child_samples": 10,
  "subsample": 0.6,
  "colsample_bytree": 0.6,
  "reg_alpha": 0.0010670537781555089,
  "reg_lambda": 0.16664067970208787,
  "random_state": 7,
}


@dataclass(frozen=True, slots=True)
class LayeredAblationCase:
  name: str
  supervised_feature_groups: tuple[str, ...] = ()
  supervised_exclude_feature_groups: tuple[str, ...] = ()
  supervised_feature_prefixes: tuple[str, ...] = ()
  supervised_exclude_feature_prefixes: tuple[str, ...] = ()
  supervised_include_feature_columns: tuple[str, ...] = ()
  supervised_exclude_feature_columns: tuple[str, ...] = ()
  meta_feature_groups: tuple[str, ...] = ()
  meta_exclude_feature_groups: tuple[str, ...] = ()
  meta_feature_prefixes: tuple[str, ...] = ()
  meta_exclude_feature_prefixes: tuple[str, ...] = ()
  meta_include_feature_columns: tuple[str, ...] = ()
  meta_exclude_feature_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LayeredFeatureAblationConfig:
  input_root: Path
  output_root: Path
  provider: str
  symbol: str
  supervised_dataset_name: str = "market_supervised"
  output_dataset_prefix: str = "market_hybrid_stacked_ablation"
  target_columns: tuple[str, ...] = tuple(build_target_columns((1, 6, 24, 96)))
  transformer: HybridTransformerConfig = HybridTransformerConfig(
    lookback=32,
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=192,
    dropout=0.1216,
    batch_size=32,
    epochs=4,
    learning_rate=0.0006168722008562335,
    weight_decay=3.0399568928322742e-05,
    validation_fraction=0.25,
    device="cpu",
    horizons=(1, 6, 24, 96),
    quantiles=(0.1, 0.5, 0.9),
  )
  initial_train_rows: int = 1024
  validation_rows: int = 256
  step_rows: int = 256
  embargo_rows: int = 1
  purge_rows: int = 96
  minimum_rows: int = 2048
  meta_validation_fraction: float = 0.25
  meta_params: dict[str, float | int] = field(
    default_factory=lambda: dict(DEFAULT_META_PARAMS)
  )
  fee_bps: float = 1.0
  slippage_bps: float = 0.5
  sizing: PositionSizingConfig = PositionSizingConfig(
    signal_scale_bps=5.0,
    signal_threshold_bps=0.5,
    volatility_column="rolling_return_std_5_bar",
    volatility_target=0.0015,
    volatility_floor=0.0005,
    max_abs_position=1.0,
    stop_loss_sigma=2.0,
    stop_loss_floor_bps=100.0,
    max_trade_risk_fraction=0.01,
    max_position_change=0.5,
    max_daily_loss_fraction=0.02,
    max_portfolio_drawdown_fraction=0.1,
  )
  cases: tuple[LayeredAblationCase, ...] = (
    LayeredAblationCase(
      name="baseline_core",
      supervised_feature_groups=("baseline_core",),
      meta_feature_groups=("baseline_core", "hybrid_transformer_oof"),
    ),
    LayeredAblationCase(
      name="trend_layer",
      supervised_feature_groups=("baseline_core", "trend_layer"),
      meta_feature_groups=("baseline_core", "trend_layer", "hybrid_transformer_oof"),
    ),
    LayeredAblationCase(
      name="structure_pruned",
      supervised_feature_groups=("baseline_core",),
      supervised_include_feature_columns=DEFAULT_PRUNED_STRUCTURE_COLUMNS,
      meta_feature_groups=("baseline_core", "hybrid_transformer_oof"),
      meta_include_feature_columns=DEFAULT_PRUNED_STRUCTURE_COLUMNS,
    ),
    LayeredAblationCase(
      name="trend_structure_pruned",
      supervised_feature_groups=("baseline_core", "trend_layer"),
      supervised_include_feature_columns=DEFAULT_PRUNED_STRUCTURE_COLUMNS,
      meta_feature_groups=("baseline_core", "trend_layer", "hybrid_transformer_oof"),
      meta_include_feature_columns=DEFAULT_PRUNED_STRUCTURE_COLUMNS,
    ),
  )


@dataclass(slots=True)
class LayeredAblationCaseSummary:
  case_name: str
  stacked_dataset_name: str
  walk_forward_directional_accuracy: float | None
  walk_forward_rmse: float | None
  meta_directional_accuracy: float | None
  meta_rmse: float | None
  net_return: float | None
  max_drawdown: float | None
  trades: int | None
  model_dir: str
  backtest_dir: str


@dataclass(slots=True)
class LayeredFeatureAblationSummary:
  run_id: str
  provider: str
  symbol: str
  input_root: str
  output_root: str
  cases: list[LayeredAblationCaseSummary]
  best_case_by_net_return: str | None


def run_layered_feature_ablation(
  config: LayeredFeatureAblationConfig,
) -> LayeredFeatureAblationSummary:
  run_id = utc_run_slug()
  root = config.output_root / run_id
  root.mkdir(parents=True, exist_ok=True)

  case_summaries: list[LayeredAblationCaseSummary] = []
  for case in config.cases:
    case_root = root / case.name
    model_root = case_root / "models"
    reports_root = case_root / "reports"

    walk_forward_summary = evaluate_hybrid_transformer_walk_forward(
      HybridTransformerWalkForwardConfig(
        input_root=config.input_root,
        output_root=model_root,
        providers=(config.provider,),
        symbols=(config.symbol,),
        dataset_name=config.supervised_dataset_name,
        target_columns=config.target_columns,
        feature_groups=case.supervised_feature_groups,
        exclude_feature_groups=case.supervised_exclude_feature_groups,
        feature_prefixes=case.supervised_feature_prefixes,
        exclude_feature_prefixes=case.supervised_exclude_feature_prefixes,
        include_feature_columns=case.supervised_include_feature_columns,
        exclude_feature_columns=case.supervised_exclude_feature_columns,
        initial_train_rows=config.initial_train_rows,
        validation_rows=config.validation_rows,
        step_rows=config.step_rows,
        embargo_rows=config.embargo_rows,
        purge_rows=config.purge_rows,
        minimum_rows=config.minimum_rows,
        transformer=config.transformer,
      )
    )
    if walk_forward_summary.runs_completed < 1:
      raise RuntimeError(f"Hybrid walk-forward produced no completed runs for case {case.name}.")

    prediction_columns = tuple(build_prediction_columns(list(config.target_columns), config.transformer.quantiles))
    stacked_dataset_name = f"{config.output_dataset_prefix}_{case.name}"
    stacked_summary = build_stacked_datasets(
      StackedDatasetConfig(
        input_root=config.input_root,
        prediction_root=model_root / "walk_forward",
        output_root=config.input_root,
        input_dataset_name=config.supervised_dataset_name,
        output_dataset_name=stacked_dataset_name,
        providers=(config.provider,),
        symbols=(config.symbol,),
        prediction_sources=(
          PredictionSource(
            family="hybrid_transformers",
            model_name="quantile_forecaster",
            alias="hybrid_transformer",
            source_columns=prediction_columns,
          ),
        ),
      )
    )
    if stacked_summary.stacked_rows_written < config.minimum_rows:
      raise RuntimeError(
        f"Stacked dataset for case {case.name} is too small: {stacked_summary.stacked_rows_written}"
      )

    meta_summary = train_meta_models(
      MetaTrainingConfig(
        input_root=config.input_root,
        output_root=model_root,
        providers=(config.provider,),
        symbols=(config.symbol,),
        model_names=("lightgbm_hybrid_return_meta",),
        validation_fraction=config.meta_validation_fraction,
        minimum_rows=config.minimum_rows,
        dataset_name=stacked_dataset_name,
        model_params={"lightgbm_hybrid_return_meta": dict(config.meta_params)},
        feature_groups=case.meta_feature_groups,
        exclude_feature_groups=case.meta_exclude_feature_groups,
        feature_prefixes=case.meta_feature_prefixes,
        exclude_feature_prefixes=case.meta_exclude_feature_prefixes,
        include_feature_columns=case.meta_include_feature_columns,
        exclude_feature_columns=case.meta_exclude_feature_columns,
      )
    )
    if meta_summary.runs_completed < 1:
      raise RuntimeError(f"Meta-model training produced no completed runs for case {case.name}.")

    model_dir = _latest_run_dir(
      model_root / "ensembles" / "lightgbm_hybrid_return_meta" / config.provider / config.symbol
    )
    backtest_summary = run_model_backtest(
      ModelBacktestConfig(
        artifact_dir=model_dir,
        input_root=config.input_root,
        output_root=reports_root / "backtests",
        dataset_name=stacked_dataset_name,
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
        sizing=config.sizing,
      )
    )
    if backtest_summary.output_dir is None:
      raise RuntimeError(f"Backtest did not produce an output directory for case {case.name}.")

    walk_forward_payload = _load_json(
      _latest_run_dir(
        model_root / "walk_forward" / "hybrid_transformers" / "quantile_forecaster" / config.provider / config.symbol
      )
      / "summary.json"
    )
    meta_payload = _load_json(model_dir / "metadata.json")
    case_summaries.append(
      LayeredAblationCaseSummary(
        case_name=case.name,
        stacked_dataset_name=stacked_dataset_name,
        walk_forward_directional_accuracy=walk_forward_payload.get("aggregate_metrics", {}).get(
          "overall_validation_forward_return_1_bar_directional_accuracy"
        ),
        walk_forward_rmse=walk_forward_payload.get("aggregate_metrics", {}).get(
          "overall_validation_forward_return_1_bar_rmse"
        ),
        meta_directional_accuracy=meta_payload.get("metrics", {}).get("validation_directional_accuracy"),
        meta_rmse=meta_payload.get("metrics", {}).get("validation_rmse"),
        net_return=backtest_summary.total_net_return,
        max_drawdown=backtest_summary.max_drawdown,
        trades=backtest_summary.trades,
        model_dir=str(model_dir),
        backtest_dir=str(backtest_summary.output_dir),
      )
    )

  best_case = max(
    case_summaries,
    key=lambda row: row.net_return if row.net_return is not None else float("-inf"),
    default=None,
  )
  summary = LayeredFeatureAblationSummary(
    run_id=run_id,
    provider=config.provider,
    symbol=config.symbol,
    input_root=str(config.input_root),
    output_root=str(root),
    cases=case_summaries,
    best_case_by_net_return=best_case.case_name if best_case else None,
  )
  (root / "summary.json").write_text(
    json.dumps(asdict(summary), indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return summary


def _latest_run_dir(root: Path) -> Path:
  if not root.exists():
    raise FileNotFoundError(root)
  run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
  if not run_dirs:
    raise FileNotFoundError(root)
  return run_dirs[-1]


def _load_json(path: Path) -> dict[str, object]:
  return json.loads(path.read_text(encoding="utf-8"))
