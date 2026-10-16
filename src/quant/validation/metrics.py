from __future__ import annotations

import math


def task_metrics(
  task_type: str,
  targets: list[float | int],
  predictions: list[float],
  *,
  prefix: str = "",
) -> dict[str, float]:
  if not targets or not predictions:
    return {}

  if task_type == "regression":
    return {
      f"{prefix}mae": mae(targets, predictions),
      f"{prefix}rmse": rmse(targets, predictions),
      f"{prefix}directional_accuracy": directional_accuracy(targets, predictions),
    }

  return {
    f"{prefix}accuracy": accuracy(targets, predictions),
    f"{prefix}macro_recall": macro_recall(targets, predictions),
  }


def mae(targets: list[float | int], predictions: list[float]) -> float:
  return sum(abs(float(target) - float(prediction)) for target, prediction in zip(targets, predictions)) / len(targets)


def rmse(targets: list[float | int], predictions: list[float]) -> float:
  return math.sqrt(sum((float(target) - float(prediction)) ** 2 for target, prediction in zip(targets, predictions)) / len(targets))


def directional_accuracy(targets: list[float | int], predictions: list[float]) -> float:
  return sum(
    1
    for target, prediction in zip(targets, predictions)
    if sign(float(target)) == sign(float(prediction))
  ) / len(targets)


def accuracy(targets: list[float | int], predictions: list[float]) -> float:
  rounded = [int(round(prediction)) for prediction in predictions]
  return sum(1 for target, prediction in zip(targets, rounded) if int(target) == prediction) / len(targets)


def macro_recall(targets: list[float | int], predictions: list[float]) -> float:
  rounded = [int(round(prediction)) for prediction in predictions]
  labels = sorted(set(int(target) for target in targets))
  recalls: list[float] = []
  for label in labels:
    positives = [prediction for target, prediction in zip(targets, rounded) if int(target) == label]
    if not positives:
      continue
    recalls.append(sum(1 for prediction in positives if prediction == label) / len(positives))
  return sum(recalls) / len(recalls) if recalls else 0.0


def sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0
