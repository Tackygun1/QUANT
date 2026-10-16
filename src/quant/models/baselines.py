from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from quant.models.base import ModelDependencyError


class _SklearnModelMixin:
  task_type: str
  model_name: str

  def __init__(self) -> None:
    self.model: Any | None = None

  def fit(self, features: list[list[float]], targets: list[float | int]) -> None:
    np, _sklearn = _require_numpy_and_sklearn()
    self.model = self._build_model()
    self.model.fit(np.asarray(features, dtype=float), np.asarray(targets))

  def predict(self, features: list[list[float]]) -> list[float]:
    np, _sklearn = _require_numpy_and_sklearn()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    predictions = self.model.predict(np.asarray(features, dtype=float))
    return predictions.tolist()

  def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
    np, _sklearn = _require_numpy_and_sklearn()
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
  def load(cls, path: Path) -> "_SklearnModelMixin":
    instance = cls()
    with path.open("rb") as handle:
      instance.model = pickle.load(handle)
    return instance

  def _build_model(self) -> Any:
    raise NotImplementedError


class RidgeReturnRegressor(_SklearnModelMixin):
  model_name = "ridge_return"
  task_type = "regression"

  def __init__(self, *, alpha: float = 1.0) -> None:
    super().__init__()
    self.alpha = alpha

  def _build_model(self) -> Any:
    _, sklearn = _require_numpy_and_sklearn()
    return sklearn["pipeline"].make_pipeline(
      sklearn["preprocessing"].StandardScaler(),
      sklearn["linear_model"].Ridge(alpha=self.alpha),
    )


class LogisticDirectionClassifier(_SklearnModelMixin):
  model_name = "logistic_direction"
  task_type = "classification"

  def __init__(self, *, max_iter: int = 1000, class_weight: str | None = "balanced") -> None:
    super().__init__()
    self.max_iter = max_iter
    self.class_weight = class_weight

  def _build_model(self) -> Any:
    _, sklearn = _require_numpy_and_sklearn()
    return sklearn["pipeline"].make_pipeline(
      sklearn["preprocessing"].StandardScaler(),
      sklearn["linear_model"].LogisticRegression(
        max_iter=self.max_iter,
        class_weight=self.class_weight,
      ),
    )


def _require_numpy_and_sklearn() -> tuple[Any, dict[str, Any]]:
  try:
    import numpy as np
  except ModuleNotFoundError as exc:
    raise ModelDependencyError(
      "numpy is required for model training. Install the 'ml' extra or add numpy."
    ) from exc

  try:
    from sklearn import linear_model, pipeline, preprocessing
  except ModuleNotFoundError as exc:
    raise ModelDependencyError(
      "scikit-learn is required for baseline model training. Install the 'ml' extra."
    ) from exc

  return np, {
    "linear_model": linear_model,
    "pipeline": pipeline,
    "preprocessing": preprocessing,
  }
