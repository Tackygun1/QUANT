from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TemperatureScaler:
  temperature: float = 1.0
  classes: tuple[int, ...] = ()

  def fit(
    self,
    probabilities: list[list[float]],
    targets: list[float | int],
    *,
    classes: list[int] | tuple[int, ...],
  ) -> None:
    np = _require_numpy()
    if not probabilities or not targets:
      self.temperature = 1.0
      self.classes = tuple(int(label) for label in classes)
      return

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(targets):
      raise ValueError("probabilities and targets must align for temperature calibration.")

    class_labels = tuple(int(label) for label in classes)
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    target_indices = [class_to_index.get(int(target), -1) for target in targets]
    valid_indices = [index for index, target_index in enumerate(target_indices) if target_index >= 0]
    if not valid_indices:
      self.temperature = 1.0
      self.classes = class_labels
      return

    matrix = matrix[valid_indices]
    y_indices = np.asarray([target_indices[index] for index in valid_indices], dtype=int)
    if len(set(int(index) for index in y_indices.tolist())) < 2:
      self.temperature = 1.0
      self.classes = class_labels
      return

    search_values = self._temperature_grid(np)
    best_temperature = 1.0
    best_loss = self._negative_log_likelihood(matrix, y_indices, temperature=best_temperature, np=np)
    for candidate in search_values:
      loss = self._negative_log_likelihood(matrix, y_indices, temperature=float(candidate), np=np)
      if loss < best_loss:
        best_loss = loss
        best_temperature = float(candidate)

    lower_bound = max(0.05, best_temperature / 1.5)
    upper_bound = min(10.0, best_temperature * 1.5)
    for candidate in np.linspace(lower_bound, upper_bound, num=31):
      loss = self._negative_log_likelihood(matrix, y_indices, temperature=float(candidate), np=np)
      if loss < best_loss:
        best_loss = loss
        best_temperature = float(candidate)

    self.temperature = max(0.05, min(10.0, float(best_temperature)))
    self.classes = class_labels

  def predict_proba(self, probabilities: list[list[float]]) -> list[list[float]]:
    np = _require_numpy()
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.size == 0:
      return matrix.tolist()
    if matrix.ndim != 2:
      raise ValueError("probabilities must be a 2D matrix for temperature calibration.")
    return self._apply_temperature(matrix, temperature=self.temperature, np=np).tolist()

  @staticmethod
  def _temperature_grid(np: object) -> list[float]:
    values = np.geomspace(0.25, 4.0, num=25)
    return [float(value) for value in values.tolist()]

  def _negative_log_likelihood(
    self,
    probabilities: object,
    target_indices: object,
    *,
    temperature: float,
    np: object,
  ) -> float:
    calibrated = self._apply_temperature(probabilities, temperature=temperature, np=np)
    likelihoods = calibrated[np.arange(len(target_indices)), target_indices]
    clipped = np.clip(likelihoods, 1e-9, 1.0)
    return float(-np.mean(np.log(clipped)))

  def _apply_temperature(self, probabilities: object, *, temperature: float, np: object) -> object:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    logits = np.log(clipped)
    scaled_logits = logits / max(temperature, 1e-9)
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    normalizer = np.sum(exp_logits, axis=1, keepdims=True)
    normalizer = np.where(normalizer <= 0, 1.0, normalizer)
    return exp_logits / normalizer


def _require_numpy() -> object:
  try:
    import numpy as np
  except ModuleNotFoundError as exc:
    from quant.models.base import ModelDependencyError

    raise ModelDependencyError(
      "numpy is required for probability calibration."
    ) from exc
  return np
