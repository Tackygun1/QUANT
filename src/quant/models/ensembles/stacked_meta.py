from __future__ import annotations

from typing import Any

from quant.models.trees.lightgbm import (
  LightGBMDirectionClassifier,
  LightGBMReturnRegressor,
)


class StackedReturnMetaModel(LightGBMReturnRegressor):
  model_name = "lightgbm_return_meta"


class StackedDirectionMetaModel(LightGBMDirectionClassifier):
  model_name = "lightgbm_direction_meta"


class HybridStackedReturnMetaModel(LightGBMReturnRegressor):
  model_name = "lightgbm_hybrid_return_meta"


class HybridStackedDirectionMetaModel(LightGBMDirectionClassifier):
  model_name = "lightgbm_hybrid_direction_meta"


class HybridStackedTradeabilityMetaModel(LightGBMDirectionClassifier):
  model_name = "lightgbm_hybrid_tradeability_meta"


_COMMON_LIGHTGBM_PARAM_KEYS = {
  "learning_rate",
  "n_estimators",
  "num_leaves",
  "max_depth",
  "min_child_samples",
  "min_child_weight",
  "min_split_gain",
  "subsample",
  "colsample_bytree",
  "feature_fraction_bynode",
  "reg_alpha",
  "reg_lambda",
  "path_smooth",
  "random_state",
  "class_weight",
  "extra_model_params",
}


def _optional_int(value: Any) -> int | None:
  if value is None or value == "":
    return None
  return int(value)


def _base_lightgbm_params(params: dict[str, Any]) -> dict[str, Any]:
  extra_model_params = dict(params.get("extra_model_params", {}))
  for key, value in params.items():
    if key not in _COMMON_LIGHTGBM_PARAM_KEYS:
      extra_model_params.setdefault(key, value)
  return {
    "learning_rate": float(params.get("learning_rate", 0.05)),
    "n_estimators": int(params.get("n_estimators", 200)),
    "num_leaves": int(params.get("num_leaves", 31)),
    "max_depth": _optional_int(params.get("max_depth")),
    "min_child_samples": int(params.get("min_child_samples", 20)),
    "min_child_weight": float(params.get("min_child_weight", 1e-3)),
    "min_split_gain": float(params.get("min_split_gain", 0.0)),
    "subsample": float(params.get("subsample", 1.0)),
    "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
    "feature_fraction_bynode": float(params.get("feature_fraction_bynode", 1.0)),
    "reg_alpha": float(params.get("reg_alpha", 0.0)),
    "reg_lambda": float(params.get("reg_lambda", 0.0)),
    "path_smooth": float(params.get("path_smooth", 0.0)),
    "random_state": int(params.get("random_state", 7)),
    "extra_model_params": extra_model_params,
  }


STACKED_META_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
  "lightgbm_return_meta": {
    "builder": lambda params: StackedReturnMetaModel(**_base_lightgbm_params(params)),
    "target_column": "forward_return_1_bar",
    "task_type": "regression",
  },
  "lightgbm_direction_meta": {
    "builder": lambda params: StackedDirectionMetaModel(
      **_base_lightgbm_params(params),
      class_weight=params.get("class_weight", "balanced"),
    ),
    "target_column": "direction_label_1_bar",
    "task_type": "classification",
  },
  "lightgbm_hybrid_return_meta": {
    "builder": lambda params: HybridStackedReturnMetaModel(**_base_lightgbm_params(params)),
    "target_column": "forward_return_1_bar",
    "task_type": "regression",
  },
  "lightgbm_hybrid_direction_meta": {
    "builder": lambda params: HybridStackedDirectionMetaModel(
      **_base_lightgbm_params(params),
      class_weight=params.get("class_weight", "balanced"),
    ),
    "target_column": "direction_label_1_bar",
    "task_type": "classification",
  },
  "lightgbm_hybrid_tradeability_meta": {
    "builder": lambda params: HybridStackedTradeabilityMetaModel(
      **_base_lightgbm_params(params),
      class_weight=params.get("class_weight", "balanced"),
    ),
    "target_column": "tradeable_label_1_bar",
    "task_type": "classification",
  },
}
