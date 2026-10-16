from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from quant.models.base import ModelDependencyError
from quant.models.probability_calibration import TemperatureScaler


class _LightGBMModelMixin:
  task_type: str
  model_name: str

  def __init__(self) -> None:
    self.model: Any | None = None
    self.probability_calibrator: TemperatureScaler | None = None

  def fit(
    self,
    features: list[list[float]],
    targets: list[float | int],
    sample_weights: list[float] | None = None,
  ) -> None:
    np, lightgbm = _require_numpy_and_lightgbm()
    self.model = self._build_model(lightgbm)
    fit_kwargs: dict[str, Any] = {}
    if sample_weights:
      fit_kwargs["sample_weight"] = np.asarray(sample_weights, dtype=float)
    self.model.fit(
      np.asarray(features, dtype=float),
      np.asarray(targets),
      **fit_kwargs,
    )

  def predict(self, features: list[list[float]]) -> list[float]:
    np, _lightgbm = _require_numpy_and_lightgbm()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    predictions = self.model.predict(np.asarray(features, dtype=float))
    return predictions.tolist()

  def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
    np, _lightgbm = _require_numpy_and_lightgbm()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    if not hasattr(self.model, "predict_proba"):
      raise RuntimeError(f"{self.model_name} does not expose predict_proba.")
    probabilities = self.model.predict_proba(np.asarray(features, dtype=float))
    rows = probabilities.tolist()
    if self.probability_calibrator is not None:
      return self.probability_calibrator.predict_proba(rows)
    return rows

  def fit_temperature_calibrator(
    self,
    features: list[list[float]],
    targets: list[float | int],
  ) -> dict[str, Any]:
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    raw_probabilities = self._predict_raw_proba(features)
    classes = getattr(self.model, "classes_", ())
    calibrator = TemperatureScaler()
    calibrator.fit(raw_probabilities, targets, classes=tuple(int(label) for label in classes))
    self.probability_calibrator = calibrator
    return {
      "method": "temperature",
      "temperature": calibrator.temperature,
      "rows": len(features),
      "classes": list(calibrator.classes),
    }

  def save(self, path: Path) -> None:
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
      pickle.dump(
        {
          "model": self.model,
          "probability_calibrator": self.probability_calibrator,
        },
        handle,
      )

  @classmethod
  def load(cls, path: Path) -> "_LightGBMModelMixin":
    instance = cls()
    with path.open("rb") as handle:
      payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload:
      instance.model = payload.get("model")
      calibrator = payload.get("probability_calibrator")
      instance.probability_calibrator = calibrator if isinstance(calibrator, TemperatureScaler) else None
      return instance
    instance.model = payload
    return instance

  def _predict_raw_proba(self, features: list[list[float]]) -> list[list[float]]:
    np, _lightgbm = _require_numpy_and_lightgbm()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    if not hasattr(self.model, "predict_proba"):
      raise RuntimeError(f"{self.model_name} does not expose predict_proba.")
    probabilities = self.model.predict_proba(np.asarray(features, dtype=float))
    return probabilities.tolist()

  def _build_model(self, lightgbm: Any) -> Any:
    raise NotImplementedError


class LightGBMReturnRegressor(_LightGBMModelMixin):
  model_name = "lightgbm_return"
  task_type = "regression"

  def __init__(
    self,
    *,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    num_leaves: int = 31,
    max_depth: int | None = None,
    min_child_samples: int = 20,
    min_child_weight: float = 1e-3,
    min_split_gain: float = 0.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    feature_fraction_bynode: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    path_smooth: float = 0.0,
    random_state: int = 7,
    extra_model_params: dict[str, Any] | None = None,
  ) -> None:
    super().__init__()
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.num_leaves = num_leaves
    self.max_depth = max_depth
    self.min_child_samples = min_child_samples
    self.min_child_weight = min_child_weight
    self.min_split_gain = min_split_gain
    self.subsample = subsample
    self.colsample_bytree = colsample_bytree
    self.feature_fraction_bynode = feature_fraction_bynode
    self.reg_alpha = reg_alpha
    self.reg_lambda = reg_lambda
    self.path_smooth = path_smooth
    self.random_state = random_state
    self.extra_model_params = dict(extra_model_params or {})

  def _build_model(self, lightgbm: Any) -> Any:
    return lightgbm.LGBMRegressor(
      learning_rate=self.learning_rate,
      n_estimators=self.n_estimators,
      num_leaves=self.num_leaves,
      max_depth=-1 if self.max_depth is None else self.max_depth,
      min_child_samples=self.min_child_samples,
      min_child_weight=self.min_child_weight,
      min_split_gain=self.min_split_gain,
      subsample=self.subsample,
      colsample_bytree=self.colsample_bytree,
      feature_fraction_bynode=self.feature_fraction_bynode,
      reg_alpha=self.reg_alpha,
      reg_lambda=self.reg_lambda,
      path_smooth=self.path_smooth,
      random_state=self.random_state,
      verbose=-1,
      **self.extra_model_params,
    )


class LightGBMDirectionClassifier(_LightGBMModelMixin):
  model_name = "lightgbm_direction"
  task_type = "classification"

  def __init__(
    self,
    *,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    num_leaves: int = 31,
    max_depth: int | None = None,
    min_child_samples: int = 20,
    min_child_weight: float = 1e-3,
    min_split_gain: float = 0.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    feature_fraction_bynode: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    path_smooth: float = 0.0,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 7,
    extra_model_params: dict[str, Any] | None = None,
  ) -> None:
    super().__init__()
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.num_leaves = num_leaves
    self.max_depth = max_depth
    self.min_child_samples = min_child_samples
    self.min_child_weight = min_child_weight
    self.min_split_gain = min_split_gain
    self.subsample = subsample
    self.colsample_bytree = colsample_bytree
    self.feature_fraction_bynode = feature_fraction_bynode
    self.reg_alpha = reg_alpha
    self.reg_lambda = reg_lambda
    self.path_smooth = path_smooth
    self.class_weight = class_weight
    self.random_state = random_state
    self.extra_model_params = dict(extra_model_params or {})

  def _build_model(self, lightgbm: Any) -> Any:
    return lightgbm.LGBMClassifier(
      learning_rate=self.learning_rate,
      n_estimators=self.n_estimators,
      num_leaves=self.num_leaves,
      max_depth=-1 if self.max_depth is None else self.max_depth,
      min_child_samples=self.min_child_samples,
      min_child_weight=self.min_child_weight,
      min_split_gain=self.min_split_gain,
      subsample=self.subsample,
      colsample_bytree=self.colsample_bytree,
      feature_fraction_bynode=self.feature_fraction_bynode,
      reg_alpha=self.reg_alpha,
      reg_lambda=self.reg_lambda,
      path_smooth=self.path_smooth,
      class_weight=self.class_weight,
      random_state=self.random_state,
      verbose=-1,
      **self.extra_model_params,
    )


def _require_numpy_and_lightgbm() -> tuple[Any, Any]:
  try:
    import numpy as np
  except ModuleNotFoundError as exc:
    raise ModelDependencyError(
      "numpy is required for LightGBM model training. Install the 'stack' extra or add numpy."
    ) from exc

  try:
    import lightgbm
  except ModuleNotFoundError as exc:
    raise ModelDependencyError(
      "lightgbm is required for meta-model training. Install the 'stack' extra."
    ) from exc

  return np, lightgbm
