from __future__ import annotations

from typing import Any


DIRECTION_TARGET_PREFIX = "direction_label"
TRADEABILITY_TARGET_PREFIX = "tradeable_label"
TREND_STATE_TARGET_PREFIX = "future_trend_state_label"
TRIPLE_BARRIER_TARGET_PREFIX = "triple_barrier_label"
PREDICTION_MARGIN_COLUMN = "prediction_margin"
DIRECTION_PROBABILITY_COLUMNS = (
  "prediction_probability_short",
  "prediction_probability_flat",
  "prediction_probability_long",
)
TRADEABILITY_PROBABILITY_COLUMNS = (
  "prediction_probability_not_tradeable",
  "prediction_probability_tradeable",
)
TRADEABILITY_MARGIN_COLUMN = "prediction_tradeability_margin"
TREND_STATE_PROBABILITY_COLUMNS = (
  "prediction_probability_downtrend",
  "prediction_probability_chop",
  "prediction_probability_uptrend",
)
TREND_STATE_MARGIN_COLUMN = "prediction_trend_state_margin"
TRIPLE_BARRIER_PROBABILITY_COLUMNS = (
  "prediction_probability_stop",
  "prediction_probability_no_hit",
  "prediction_probability_profit",
)
TRIPLE_BARRIER_MARGIN_COLUMN = "prediction_triple_barrier_margin"
DIRECTION_CLASSIFICATION_OUTPUT_COLUMNS = (
  PREDICTION_MARGIN_COLUMN,
  *DIRECTION_PROBABILITY_COLUMNS,
  "prediction_probability_max",
  "prediction_confidence",
)


def is_direction_classification(*, task_type: str, target_column: str) -> bool:
  return task_type == "classification" and target_column.startswith(DIRECTION_TARGET_PREFIX)


def is_supported_ternary_classification(*, task_type: str, target_column: str) -> bool:
  return task_type == "classification" and _classification_spec(target_column) is not None


def direction_probability_metadata() -> dict[str, Any]:
  return {
    "primary_prediction_column": PREDICTION_MARGIN_COLUMN,
    "prediction_output_type": "classification_probability_margin",
    "hard_prediction_column": "prediction",
    "probability_columns": list(DIRECTION_PROBABILITY_COLUMNS),
    "confidence_column": "prediction_confidence",
  }


def classification_probability_metadata(*, target_column: str) -> dict[str, Any]:
  spec = _classification_spec(target_column)
  if spec is None:
    return {}
  return {
    "primary_prediction_column": spec["margin_column"],
    "prediction_output_type": f"{spec['name']}_classification_probability_margin",
    "hard_prediction_column": "prediction",
    "probability_columns": list(spec["probability_columns"]),
    "confidence_column": "prediction_confidence",
  }


def build_direction_prediction_outputs(
  *,
  model: Any,
  features: list[list[float]],
  predictions: list[float] | None = None,
) -> list[dict[str, float]]:
  return build_classification_prediction_outputs(
    model=model,
    features=features,
    target_column=DIRECTION_TARGET_PREFIX,
    predictions=predictions,
  )


def build_classification_prediction_outputs(
  *,
  model: Any,
  features: list[list[float]],
  target_column: str,
  predictions: list[float] | None = None,
) -> list[dict[str, float]]:
  spec = _classification_spec(target_column)
  if spec is None:
    hard_predictions = predictions if predictions is not None else model.predict(features)
    return [{"prediction": float(prediction)} for prediction in hard_predictions]

  hard_predictions = list(predictions) if predictions is not None else [float(value) for value in model.predict(features)]
  rows = [{"prediction": float(prediction)} for prediction in hard_predictions]
  if not features or not hasattr(model, "predict_proba"):
    return rows

  try:
    probability_rows = model.predict_proba(features)
  except RuntimeError:
    return rows

  classes = _model_classes(model)
  if not classes or len(classes) != len(probability_rows[0]):
    return rows

  output_rows: list[dict[str, float]] = []
  for hard_prediction, probabilities in zip(hard_predictions, probability_rows):
    probability_by_class = {
      label: float(probability)
      for label, probability in zip(classes, probabilities)
    }
    sorted_probabilities = sorted((float(value) for value in probabilities), reverse=True)
    top_probability = sorted_probabilities[0] if sorted_probabilities else 0.0
    runner_up_probability = sorted_probabilities[1] if len(sorted_probabilities) > 1 else 0.0
    probability_columns = spec["probability_columns"]
    class_labels = spec["classes"]
    output = {
      "prediction": float(hard_prediction),
      "prediction_probability_max": top_probability,
      "prediction_confidence": max(0.0, top_probability - runner_up_probability),
    }
    for label, column in zip(class_labels, probability_columns):
      output[column] = probability_by_class.get(label, 0.0)
    positive_probability = probability_by_class.get(spec["margin_positive_class"], 0.0)
    negative_probability = probability_by_class.get(spec["margin_negative_class"], 0.0)
    output[spec["margin_column"]] = positive_probability - negative_probability
    calibrated_prediction = float(hard_prediction)
    if probability_by_class:
      calibrated_prediction = float(max(probability_by_class, key=probability_by_class.get))
    output["prediction"] = calibrated_prediction
    output_rows.append(output)
  return output_rows


def _classification_spec(target_column: str) -> dict[str, Any] | None:
  if target_column.startswith(DIRECTION_TARGET_PREFIX):
    return {
      "name": "direction",
      "margin_column": PREDICTION_MARGIN_COLUMN,
      "probability_columns": DIRECTION_PROBABILITY_COLUMNS,
      "classes": (-1, 0, 1),
      "margin_positive_class": 1,
      "margin_negative_class": -1,
    }
  if target_column.startswith(TRADEABILITY_TARGET_PREFIX):
    return {
      "name": "tradeability",
      "margin_column": TRADEABILITY_MARGIN_COLUMN,
      "probability_columns": TRADEABILITY_PROBABILITY_COLUMNS,
      "classes": (0, 1),
      "margin_positive_class": 1,
      "margin_negative_class": 0,
    }
  if target_column.startswith(TREND_STATE_TARGET_PREFIX):
    return {
      "name": "trend_state",
      "margin_column": TREND_STATE_MARGIN_COLUMN,
      "probability_columns": TREND_STATE_PROBABILITY_COLUMNS,
      "classes": (-1, 0, 1),
      "margin_positive_class": 1,
      "margin_negative_class": -1,
    }
  if target_column.startswith(TRIPLE_BARRIER_TARGET_PREFIX):
    return {
      "name": "triple_barrier",
      "margin_column": TRIPLE_BARRIER_MARGIN_COLUMN,
      "probability_columns": TRIPLE_BARRIER_PROBABILITY_COLUMNS,
      "classes": (-1, 0, 1),
      "margin_positive_class": 1,
      "margin_negative_class": -1,
    }
  return None


def _model_classes(model: Any) -> list[int]:
  native_model = getattr(model, "model", model)
  classes = getattr(native_model, "classes_", None)
  if classes is None and hasattr(native_model, "steps"):
    classes = getattr(native_model.steps[-1][1], "classes_", None)
  if classes is None:
    return []
  labels: list[int] = []
  for label in list(classes):
    try:
      labels.append(int(label))
    except (TypeError, ValueError):
      try:
        labels.append(int(float(label)))
      except (TypeError, ValueError):
        return []
  return labels
