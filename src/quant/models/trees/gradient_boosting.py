from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from quant.models.base import ModelDependencyError


class _TreeModelMixin:
  model_name: str
  task_type: str

  def __init__(self) -> None:
    self.model: Any | None = None

  def fit(self, features: list[list[float]], targets: list[float | int]) -> None:
    np, ensemble = _require_numpy_and_sklearn_tree()
    self.model = self._build_model(ensemble)
    self.model.fit(np.asarray(features, dtype=float), np.asarray(targets))

  def predict(self, features: list[list[float]]) -> list[float]:
    np, _ensemble = _require_numpy_and_sklearn_tree()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    predictions = self.model.predict(np.asarray(features, dtype=float))
    return predictions.tolist()

  def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
    np, _ensemble = _require_numpy_and_sklearn_tree()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    if not hasattr(self.model, "predict_proba"):
      raise RuntimeError(f"{self.model_name} does not expose predict_proba.")
    probabilities = self.model.predict_proba(np.asarray(features, dtype=float))
    return probabilities.tolist()

  def save(self, path: Path) -> None:
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
      pickle.dump(self.model, handle)

  @classmethod
  def load(cls, path: Path) -> "_TreeModelMixin":
    instance = cls()
    with path.open("rb") as handle:
      instance.model = pickle.load(handle)
    return instance

  def _build_model(self, ensemble: Any) -> Any:
    raise NotImplementedError


class HistGradientBoostingReturnRegressor(_TreeModelMixin):
  model_name = "histgb_return"
  task_type = "regression"

  def __init__(
    self,
    *,
    max_depth: int | None = 4,
    learning_rate: float = 0.05,
    max_iter: int = 200,
  ) -> None:
    super().__init__()
    self.max_depth = max_depth
    self.learning_rate = learning_rate
    self.max_iter = max_iter

  def _build_model(self, ensemble: Any) -> Any:
    return ensemble.HistGradientBoostingRegressor(
      max_depth=self.max_depth,
      learning_rate=self.learning_rate,
      max_iter=self.max_iter,
    )


class HistGradientBoostingDirectionClassifier(_TreeModelMixin):
  model_name = "histgb_direction"
  task_type = "classification"

  def __init__(
    self,
    *,
    max_depth: int | None = 4,
    learning_rate: float = 0.05,
    max_iter: int = 200,
  ) -> None:
    super().__init__()
    self.max_depth = max_depth
    self.learning_rate = learning_rate
    self.max_iter = max_iter

  def _build_model(self, ensemble: Any) -> Any:
    return ensemble.HistGradientBoostingClassifier(
      max_depth=self.max_depth,
      learning_rate=self.learning_rate,
      max_iter=self.max_iter,
    )


def _require_numpy_and_sklearn_tree() -> tuple[Any, Any]:
  try:
    import numpy as np
  except ModuleNotFoundError as exc:
    raise ModelDependencyError(
      "numpy is required for tree model training. Install the 'ml' extra or add numpy."
    ) from exc

  try:
    from sklearn import ensemble
  except ModuleNotFoundError as exc:
    raise ModelDependencyError(
      "scikit-learn is required for tree model training. Install the 'ml' extra."
    ) from exc

  return np, ensemble
