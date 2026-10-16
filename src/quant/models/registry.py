from __future__ import annotations

from typing import Any

from quant.models.baselines import LogisticDirectionClassifier, RidgeReturnRegressor
from quant.models.trees.gradient_boosting import (
  HistGradientBoostingDirectionClassifier,
  HistGradientBoostingReturnRegressor,
)


TABULAR_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
  "ridge_return": {
    "builder": lambda params: RidgeReturnRegressor(alpha=float(params.get("alpha", 1.0))),
    "target_column": "forward_return_1_bar",
    "task_type": "regression",
  },
  "logistic_direction": {
    "builder": lambda params: LogisticDirectionClassifier(
      max_iter=int(params.get("max_iter", 1000)),
      class_weight=params.get("class_weight", "balanced"),
    ),
    "target_column": "direction_label_1_bar",
    "task_type": "classification",
  },
  "histgb_return": {
    "builder": lambda params: HistGradientBoostingReturnRegressor(
      max_depth=optional_int(params.get("max_depth", 4)),
      learning_rate=float(params.get("learning_rate", 0.05)),
      max_iter=int(params.get("max_iter", 200)),
    ),
    "target_column": "forward_return_1_bar",
    "task_type": "regression",
  },
  "histgb_direction": {
    "builder": lambda params: HistGradientBoostingDirectionClassifier(
      max_depth=optional_int(params.get("max_depth", 4)),
      learning_rate=float(params.get("learning_rate", 0.05)),
      max_iter=int(params.get("max_iter", 200)),
    ),
    "target_column": "direction_label_1_bar",
    "task_type": "classification",
  },
  "histgb_trend_state": {
    "builder": lambda params: HistGradientBoostingDirectionClassifier(
      max_depth=optional_int(params.get("max_depth", 3)),
      learning_rate=float(params.get("learning_rate", 0.05)),
      max_iter=int(params.get("max_iter", 150)),
    ),
    "target_column": "future_trend_state_label_24_bar",
    "task_type": "classification",
  },
  "histgb_trend_state_fast": {
    "builder": lambda params: HistGradientBoostingDirectionClassifier(
      max_depth=optional_int(params.get("max_depth", 3)),
      learning_rate=float(params.get("learning_rate", 0.05)),
      max_iter=int(params.get("max_iter", 150)),
    ),
    "target_column": "future_trend_state_label_6_bar",
    "task_type": "classification",
  },
  "histgb_triple_barrier_fast": {
    "builder": lambda params: HistGradientBoostingDirectionClassifier(
      max_depth=optional_int(params.get("max_depth", 3)),
      learning_rate=float(params.get("learning_rate", 0.05)),
      max_iter=int(params.get("max_iter", 150)),
    ),
    "target_column": "triple_barrier_label_6_bar",
    "task_type": "classification",
  },
}


def optional_int(value: Any) -> int | None:
  if value in (None, "", "none"):
    return None
  return int(value)
