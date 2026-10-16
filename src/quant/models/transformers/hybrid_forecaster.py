from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quant.models.base import ModelDependencyError


@dataclass(frozen=True, slots=True)
class HybridTransformerConfig:
  lookback: int = 48
  backbone: str = "timemixer"
  d_model: int = 64
  nhead: int = 4
  num_layers: int = 2
  dim_feedforward: int = 128
  dropout: float = 0.1
  batch_size: int = 32
  epochs: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 1e-4
  validation_fraction: float = 0.25
  device: str = "cpu"
  horizons: tuple[int, ...] = (1, 6, 24, 96)
  quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
  scales: tuple[int, ...] = (1, 2, 4)
  mixer_layers: int = 2
  auxiliary_loss_weight: float = 0.25
  auxiliary_loss_weights: dict[str, float] = field(default_factory=dict)


class HybridQuantileTransformer:
  model_name = "hybrid_quantile_transformer"
  task_type = "regression"

  def __init__(self, config: HybridTransformerConfig | None = None) -> None:
    self.config = config or HybridTransformerConfig()
    self.model: Any | None = None
    self.feature_names: list[str] = []
    self.target_columns: list[str] = []
    self.feature_means: list[float] = []
    self.feature_stds: list[float] = []
    self.target_means: list[float] = []
    self.target_stds: list[float] = []
    self.auxiliary_target_columns: list[str] = []
    self.auxiliary_class_values: list[list[int]] = []

  def fit(
    self,
    sequences: list[list[list[float]]],
    targets: list[list[float]],
    *,
    feature_names: list[str],
    target_columns: list[str],
    auxiliary_targets: list[list[int]] | None = None,
    auxiliary_target_columns: list[str] | None = None,
    validation_sequences: list[list[list[float]]] | None = None,
    validation_targets: list[list[float]] | None = None,
    validation_auxiliary_targets: list[list[int]] | None = None,
    sample_weights: list[float] | None = None,
  ) -> dict[str, float]:
    torch = _require_torch()
    if not sequences:
      raise ValueError("No sequences provided for hybrid transformer training.")
    if not targets:
      raise ValueError("No targets provided for hybrid transformer training.")
    if len(target_columns) != len(targets[0]):
      raise ValueError("Target column count does not match hybrid transformer target width.")

    self.feature_names = list(feature_names)
    self.target_columns = list(target_columns)
    if sample_weights == []:
      sample_weights = None
    self.auxiliary_target_columns = list(auxiliary_target_columns or ())
    all_auxiliary_targets = _normalize_auxiliary_target_rows(
      auxiliary_targets,
      row_count=len(targets),
      column_count=len(self.auxiliary_target_columns),
    )
    if sample_weights is not None and len(sample_weights) != len(sequences):
      raise ValueError("sample_weights length must match the training sequence count.")
    if validation_sequences is None or validation_targets is None:
      split_index = _split_index(len(sequences), self.config.validation_fraction)
      train_sequences = sequences[:split_index]
      train_targets = targets[:split_index]
      train_auxiliary_targets = all_auxiliary_targets[:split_index]
      valid_sequences = sequences[split_index:]
      valid_targets = targets[split_index:]
      valid_auxiliary_targets = all_auxiliary_targets[split_index:]
      train_sample_weights = sample_weights[:split_index] if sample_weights is not None else None
    else:
      train_sequences = sequences
      train_targets = targets
      train_auxiliary_targets = all_auxiliary_targets
      valid_sequences = validation_sequences
      valid_targets = validation_targets
      valid_auxiliary_targets = _normalize_auxiliary_target_rows(
        validation_auxiliary_targets,
        row_count=len(validation_targets),
        column_count=len(self.auxiliary_target_columns),
      )
      train_sample_weights = sample_weights

    self.auxiliary_class_values = _auxiliary_class_values(
      train_auxiliary_targets,
      valid_auxiliary_targets,
      column_count=len(self.auxiliary_target_columns),
    )

    self.feature_means, self.feature_stds = _feature_stats(train_sequences)
    self.target_means, self.target_stds = _target_stats(train_targets)

    normalized_train_sequences = _normalize_sequences(
      train_sequences,
      self.feature_means,
      self.feature_stds,
    )
    normalized_train_targets = _normalize_target_matrix(
      train_targets,
      self.target_means,
      self.target_stds,
    )
    normalized_valid_sequences = _normalize_sequences(
      valid_sequences,
      self.feature_means,
      self.feature_stds,
    )
    normalized_valid_targets = _normalize_target_matrix(
      valid_targets,
      self.target_means,
      self.target_stds,
    )

    device = torch.device(self.config.device)
    self.model = _HybridSequenceRegressor(
      input_dim=len(self.feature_names),
      lookback=self.config.lookback,
      backbone=self.config.backbone,
      d_model=self.config.d_model,
      nhead=self.config.nhead,
      num_layers=self.config.num_layers,
      dim_feedforward=self.config.dim_feedforward,
      dropout=self.config.dropout,
      output_dim=self._model_output_dim(),
      scales=self.config.scales,
      mixer_layers=self.config.mixer_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
      self.model.parameters(),
      lr=self.config.learning_rate,
      weight_decay=self.config.weight_decay,
    )

    train_tensor = torch.tensor(normalized_train_sequences, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(normalized_train_targets, dtype=torch.float32, device=device)
    auxiliary_tensor = (
      torch.tensor(
        _auxiliary_target_indices(train_auxiliary_targets, self.auxiliary_class_values),
        dtype=torch.long,
        device=device,
      )
      if self.auxiliary_target_columns
      else None
    )
    sample_weight_tensor = (
      torch.tensor(_normalize_sample_weights(train_sample_weights), dtype=torch.float32, device=device)
      if train_sample_weights
      else None
    )

    best_validation = math.inf
    best_state = None

    for _epoch in range(self.config.epochs):
      self.model.train()
      batch_losses: list[float] = []
      for start in range(0, len(train_sequences), self.config.batch_size):
        end = start + self.config.batch_size
        batch_x = train_tensor[start:end]
        batch_y = target_tensor[start:end]
        batch_aux_y = auxiliary_tensor[start:end] if auxiliary_tensor is not None else None
        batch_weights = sample_weight_tensor[start:end] if sample_weight_tensor is not None else None
        optimizer.zero_grad()
        outputs = self.model(batch_x)
        regression_outputs, auxiliary_outputs = _split_model_outputs(
          outputs,
          regression_output_dim=self._regression_output_dim(),
        )
        predictions = regression_outputs.reshape(
          len(batch_x),
          len(self.target_columns),
          len(self.config.quantiles),
        )
        loss = _quantile_loss(
          predictions,
          batch_y,
          quantiles=self.config.quantiles,
          sample_weights=batch_weights,
        )
        if batch_aux_y is not None and auxiliary_outputs is not None:
          loss = loss + self.config.auxiliary_loss_weight * _auxiliary_classification_loss(
            auxiliary_outputs,
            batch_aux_y,
            target_columns=self.auxiliary_target_columns,
            class_values=self.auxiliary_class_values,
            loss_weights=self.config.auxiliary_loss_weights,
            sample_weights=batch_weights,
          )
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.item()))

      validation_loss = (
        self._evaluate(
          normalized_valid_sequences,
          normalized_valid_targets,
          valid_auxiliary_targets,
          device,
        )
        if normalized_valid_sequences
        else float(sum(batch_losses) / max(len(batch_losses), 1))
      )
      if validation_loss < best_validation:
        best_validation = validation_loss
        best_state = {
          key: value.detach().cpu()
          for key, value in self.model.state_dict().items()
        }

    if best_state is not None:
      self.model.load_state_dict(best_state)

    train_outputs = self.predict_output_rows(train_sequences)
    metrics = _output_metrics(
      target_columns=self.target_columns,
      targets=train_targets,
      output_rows=train_outputs,
      quantiles=self.config.quantiles,
      auxiliary_target_columns=self.auxiliary_target_columns,
      auxiliary_targets=train_auxiliary_targets,
      prefix="train_",
    )
    if valid_sequences:
      valid_outputs = self.predict_output_rows(valid_sequences)
      metrics.update(
        _output_metrics(
          target_columns=self.target_columns,
          targets=valid_targets,
          output_rows=valid_outputs,
          quantiles=self.config.quantiles,
          auxiliary_target_columns=self.auxiliary_target_columns,
          auxiliary_targets=valid_auxiliary_targets,
          prefix="validation_",
        )
      )
    return metrics

  def predict(self, sequences: list[list[list[float]]]) -> list[float]:
    primary_column = self.primary_prediction_column
    return [float(row.get(primary_column, 0.0)) for row in self.predict_output_rows(sequences)]

  def predict_output_rows(self, sequences: list[list[list[float]]]) -> list[dict[str, float]]:
    torch = _require_torch()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    if not sequences:
      return []

    normalized_sequences = _normalize_sequences(
      sequences,
      self.feature_means,
      self.feature_stds,
    )
    device = next(self.model.parameters()).device
    self.model.eval()
    with torch.no_grad():
      tensor = torch.tensor(normalized_sequences, dtype=torch.float32, device=device)
      outputs = self.model(tensor)
      regression_outputs, auxiliary_outputs = _split_model_outputs(
        outputs,
        regression_output_dim=self._regression_output_dim(),
      )
      predictions = regression_outputs.reshape(
        len(normalized_sequences),
        len(self.target_columns),
        len(self.config.quantiles),
      )
      predictions = predictions.detach().cpu().tolist()
      auxiliary_predictions = (
        auxiliary_outputs.detach().cpu().tolist()
        if auxiliary_outputs is not None and self.auxiliary_target_columns
        else []
      )

    output_rows: list[dict[str, float]] = []
    for index, sample in enumerate(predictions):
      row = _denormalize_output_row(
        sample=sample,
        target_columns=self.target_columns,
        quantiles=self.config.quantiles,
        target_means=self.target_means,
        target_stds=self.target_stds,
      )
      if auxiliary_predictions:
        row.update(
          _auxiliary_output_row(
            logits=auxiliary_predictions[index],
            target_columns=self.auxiliary_target_columns,
            class_values=self.auxiliary_class_values,
          )
        )
      output_rows.append(row)
    return output_rows

  @property
  def primary_prediction_column(self) -> str:
    return _prediction_quantile_column(self.target_columns[0], _median_quantile(self.config.quantiles))

  def save(self, path: Path) -> None:
    torch = _require_torch()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "config": asdict(self.config),
        "feature_names": self.feature_names,
        "target_columns": self.target_columns,
        "auxiliary_target_columns": self.auxiliary_target_columns,
        "auxiliary_class_values": self.auxiliary_class_values,
        "feature_means": self.feature_means,
        "feature_stds": self.feature_stds,
        "target_means": self.target_means,
        "target_stds": self.target_stds,
        "state_dict": self.model.state_dict(),
      },
      path,
    )

  @classmethod
  def load(cls, path: Path) -> "HybridQuantileTransformer":
    torch = _require_torch()
    payload = torch.load(path, map_location="cpu")
    raw_config = dict(payload["config"])
    state_dict = payload.get("state_dict", {})
    if "backbone" not in raw_config:
      raw_config["backbone"] = _infer_backbone_from_state_dict(state_dict)
    elif raw_config["backbone"] in {"timemixer", "multiscale_mixer"} and _state_dict_looks_transformer(state_dict):
      raw_config["backbone"] = "transformer"
    config = HybridTransformerConfig(**raw_config)
    instance = cls(config)
    instance.feature_names = list(payload["feature_names"])
    instance.target_columns = list(payload["target_columns"])
    instance.auxiliary_target_columns = list(payload.get("auxiliary_target_columns", []))
    instance.auxiliary_class_values = [
      [int(label) for label in labels]
      for labels in payload.get("auxiliary_class_values", [])
    ]
    instance.feature_means = [float(value) for value in payload.get("feature_means", [])]
    instance.feature_stds = [float(value) for value in payload.get("feature_stds", [])]
    instance.target_means = [float(value) for value in payload.get("target_means", [])]
    instance.target_stds = [float(value) for value in payload.get("target_stds", [])]
    instance.model = _HybridSequenceRegressor(
      input_dim=len(instance.feature_names),
      lookback=config.lookback,
      backbone=config.backbone,
      d_model=config.d_model,
      nhead=config.nhead,
      num_layers=config.num_layers,
      dim_feedforward=config.dim_feedforward,
      dropout=config.dropout,
      output_dim=instance._model_output_dim(),
      scales=config.scales,
      mixer_layers=config.mixer_layers,
    )
    instance.model.load_state_dict(payload["state_dict"])
    return instance

  def _evaluate(
    self,
    sequences: list[list[list[float]]],
    targets: list[list[float]],
    auxiliary_targets: list[list[int]],
    device: Any,
  ) -> float:
    torch = _require_torch()
    self.model.eval()
    with torch.no_grad():
      features = torch.tensor(sequences, dtype=torch.float32, device=device)
      labels = torch.tensor(targets, dtype=torch.float32, device=device)
      outputs = self.model(features)
      regression_outputs, auxiliary_outputs = _split_model_outputs(
        outputs,
        regression_output_dim=self._regression_output_dim(),
      )
      predictions = regression_outputs.reshape(
        len(sequences),
        len(self.target_columns),
        len(self.config.quantiles),
      )
      loss = _quantile_loss(predictions, labels, quantiles=self.config.quantiles)
      if self.auxiliary_target_columns and auxiliary_outputs is not None:
        auxiliary_labels = torch.tensor(
          _auxiliary_target_indices(auxiliary_targets, self.auxiliary_class_values),
          dtype=torch.long,
          device=device,
        )
        loss = loss + self.config.auxiliary_loss_weight * _auxiliary_classification_loss(
          auxiliary_outputs,
          auxiliary_labels,
          target_columns=self.auxiliary_target_columns,
          class_values=self.auxiliary_class_values,
          loss_weights=self.config.auxiliary_loss_weights,
        )
    return float(loss.item())

  def _regression_output_dim(self) -> int:
    return len(self.target_columns) * len(self.config.quantiles)

  def _auxiliary_output_dim(self) -> int:
    return sum(len(labels) for labels in self.auxiliary_class_values)

  def _model_output_dim(self) -> int:
    return self._regression_output_dim() + self._auxiliary_output_dim()


class _HybridSequenceRegressor:  # pragma: no cover - exercised only when torch is installed
  def __new__(
    cls,
    *,
    input_dim: int,
    lookback: int,
    backbone: str,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    output_dim: int,
    scales: tuple[int, ...],
    mixer_layers: int,
  ) -> Any:
    torch = _require_torch()

    if backbone == "transformer":
      return _transformer_module(
        torch=torch,
        input_dim=input_dim,
        lookback=lookback,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_dim=output_dim,
      )
    if backbone in {"timemixer", "multiscale_mixer"}:
      return _timemixer_module(
        torch=torch,
        input_dim=input_dim,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_dim=output_dim,
        scales=scales,
        mixer_layers=mixer_layers,
      )
    raise ValueError(f"Unsupported hybrid sequence backbone: {backbone}")


def _transformer_module(
  *,
  torch: Any,
  input_dim: int,
  lookback: int,
  d_model: int,
  nhead: int,
  num_layers: int,
  dim_feedforward: int,
  dropout: float,
  output_dim: int,
) -> Any:
  class Module(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.input_projection = torch.nn.Linear(input_dim, d_model)
      self.position_embedding = torch.nn.Parameter(torch.zeros(lookback, d_model))
      encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
      )
      self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
      self.head = torch.nn.Sequential(
        torch.nn.LayerNorm(d_model),
        torch.nn.Linear(d_model, d_model),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(d_model, output_dim),
      )

    def forward(self, features: Any) -> Any:
      projected = self.input_projection(features)
      positioned = projected + self.position_embedding.unsqueeze(0)
      encoded = self.encoder(positioned)
      pooled = encoded[:, -1, :]
      return self.head(pooled)

  return Module()


def _state_dict_looks_transformer(state_dict: dict[str, Any]) -> bool:
  return "position_embedding" in state_dict


def _infer_backbone_from_state_dict(state_dict: dict[str, Any]) -> str:
  if _state_dict_looks_transformer(state_dict):
    return "transformer"
  return "timemixer"


def _timemixer_module(
  *,
  torch: Any,
  input_dim: int,
  d_model: int,
  dim_feedforward: int,
  dropout: float,
  output_dim: int,
  scales: tuple[int, ...],
  mixer_layers: int,
) -> Any:
  normalized_scales = tuple(sorted({max(1, int(scale)) for scale in scales}))

  class MixerBlock(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.temporal = torch.nn.Sequential(
        torch.nn.LayerNorm(d_model),
        torch.nn.Linear(d_model, dim_feedforward),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(dim_feedforward, d_model),
      )
      self.channel = torch.nn.Sequential(
        torch.nn.LayerNorm(d_model),
        torch.nn.Linear(d_model, dim_feedforward),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(dim_feedforward, d_model),
      )

    def forward(self, values: Any) -> Any:
      mixed = values + self.temporal(values)
      trend = mixed.mean(dim=1, keepdim=True)
      seasonal = mixed - trend
      return seasonal + trend + self.channel(seasonal + trend)

  class Module(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.input_projection = torch.nn.Linear(input_dim, d_model)
      self.scale_mixers = torch.nn.ModuleList(
        torch.nn.ModuleList(MixerBlock() for _ in range(max(1, mixer_layers)))
        for _ in normalized_scales
      )
      self.scale_gate = torch.nn.Sequential(
        torch.nn.Linear(d_model * len(normalized_scales), dim_feedforward),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(dim_feedforward, d_model),
        torch.nn.GELU(),
      )
      self.head = torch.nn.Sequential(
        torch.nn.LayerNorm(d_model),
        torch.nn.Linear(d_model, dim_feedforward),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(dim_feedforward, output_dim),
      )

    def forward(self, features: Any) -> Any:
      scale_vectors = []
      for scale, blocks in zip(normalized_scales, self.scale_mixers):
        scaled = _downsample_sequence(torch, features, scale)
        projected = self.input_projection(scaled)
        for block in blocks:
          projected = block(projected)
        scale_vectors.append(projected[:, -1, :])
      fused = self.scale_gate(torch.cat(scale_vectors, dim=-1))
      return self.head(fused)

  return Module()


def _downsample_sequence(torch: Any, values: Any, scale: int) -> Any:
  if scale <= 1:
    return values
  transposed = values.transpose(1, 2)
  pooled = torch.nn.functional.avg_pool1d(
    transposed,
    kernel_size=scale,
    stride=scale,
    ceil_mode=True,
  )
  return pooled.transpose(1, 2)


def build_target_columns(horizons: tuple[int, ...]) -> list[str]:
  return [f"forward_return_{horizon}_bar" for horizon in horizons]


def build_prediction_columns(
  target_columns: list[str],
  quantiles: tuple[float, ...],
) -> list[str]:
  columns: list[str] = []
  for target_column in target_columns:
    for quantile in quantiles:
      columns.append(_prediction_quantile_column(target_column, quantile))
    if len(quantiles) >= 2:
      columns.append(_prediction_spread_column(target_column))
  return columns


def build_auxiliary_prediction_columns(
  target_columns: list[str],
  class_values: list[list[int]] | None = None,
) -> list[str]:
  columns: list[str] = []
  for index, target_column in enumerate(target_columns):
    labels = class_values[index] if class_values and index < len(class_values) else _default_auxiliary_labels(target_column)
    columns.extend(_auxiliary_prediction_columns(target_column, labels))
  return columns


def auxiliary_prediction_column(target_column: str) -> str:
  return _auxiliary_prediction_column(target_column)


def _output_metrics(
  *,
  target_columns: list[str],
  targets: list[list[float]],
  output_rows: list[dict[str, float]],
  quantiles: tuple[float, ...],
  auxiliary_target_columns: list[str],
  auxiliary_targets: list[list[int]],
  prefix: str,
) -> dict[str, float]:
  metrics: dict[str, float] = {}
  median_quantile = _median_quantile(quantiles)
  for column_index, target_column in enumerate(target_columns):
    target_values = [float(row[column_index]) for row in targets]
    prediction_values = [
      float(output_row.get(_prediction_quantile_column(target_column, median_quantile), 0.0))
      for output_row in output_rows
    ]
    suffix = target_column
    metrics[f"{prefix}mae_{suffix}"] = _mae(target_values, prediction_values)
    metrics[f"{prefix}rmse_{suffix}"] = _rmse(target_values, prediction_values)
    metrics[f"{prefix}directional_accuracy_{suffix}"] = _directional_accuracy(target_values, prediction_values)

  mae_keys = [key for key in metrics if key.startswith(prefix + "mae_")]
  rmse_keys = [key for key in metrics if key.startswith(prefix + "rmse_")]
  da_keys = [key for key in metrics if key.startswith(prefix + "directional_accuracy_")]
  if mae_keys:
    metrics[f"{prefix}mae_mean"] = sum(metrics[key] for key in mae_keys) / len(mae_keys)
  if rmse_keys:
    metrics[f"{prefix}rmse_mean"] = sum(metrics[key] for key in rmse_keys) / len(rmse_keys)
  if da_keys:
    metrics[f"{prefix}directional_accuracy_mean"] = sum(metrics[key] for key in da_keys) / len(da_keys)
  for column_index, target_column in enumerate(auxiliary_target_columns):
    target_values = [int(row[column_index]) for row in auxiliary_targets]
    prediction_column = _auxiliary_prediction_column(target_column)
    prediction_values = [
      int(round(float(output_row.get(prediction_column, 0.0))))
      for output_row in output_rows
    ]
    suffix = target_column
    metrics[f"{prefix}auxiliary_accuracy_{suffix}"] = _classification_accuracy(target_values, prediction_values)
    metrics[f"{prefix}auxiliary_macro_recall_{suffix}"] = _classification_macro_recall(target_values, prediction_values)
  return metrics


def _quantile_loss(
  predictions: Any,
  targets: Any,
  *,
  quantiles: tuple[float, ...],
  sample_weights: Any | None = None,
) -> Any:
  torch = _require_torch()
  target_tensor = targets.unsqueeze(-1).expand_as(predictions)
  errors = target_tensor - predictions
  loss_terms = []
  for quantile_index, quantile in enumerate(quantiles):
    q = torch.tensor(float(quantile), dtype=predictions.dtype, device=predictions.device)
    quantile_errors = errors[:, :, quantile_index]
    loss_terms.append(torch.maximum(q * quantile_errors, (q - 1.0) * quantile_errors))
  stacked = torch.stack(loss_terms, dim=-1)
  sample_loss = stacked.mean(dim=(1, 2))
  return _weighted_mean(sample_loss, sample_weights)


def _denormalize_output_row(
  *,
  sample: list[list[float]],
  target_columns: list[str],
  quantiles: tuple[float, ...],
  target_means: list[float],
  target_stds: list[float],
) -> dict[str, float]:
  row: dict[str, float] = {}
  for target_index, target_column in enumerate(target_columns):
    quantile_values = [
      float(sample[target_index][quantile_index]) * target_stds[target_index] + target_means[target_index]
      for quantile_index in range(len(quantiles))
    ]
    quantile_values = sorted(quantile_values)
    for quantile, value in zip(quantiles, quantile_values):
      row[_prediction_quantile_column(target_column, quantile)] = value
    if len(quantile_values) >= 2:
      row[_prediction_spread_column(target_column)] = quantile_values[-1] - quantile_values[0]
  return row


def _split_model_outputs(outputs: Any, *, regression_output_dim: int) -> tuple[Any, Any | None]:
  regression_outputs = outputs[:, :regression_output_dim]
  if outputs.shape[1] <= regression_output_dim:
    return regression_outputs, None
  return regression_outputs, outputs[:, regression_output_dim:]


def _auxiliary_class_values(
  train_targets: list[list[int]],
  validation_targets: list[list[int]],
  *,
  column_count: int,
) -> list[list[int]]:
  if column_count == 0:
    return []
  class_values: list[list[int]] = []
  all_targets = [*train_targets, *validation_targets]
  for column_index in range(column_count):
    labels = sorted({int(row[column_index]) for row in all_targets if len(row) > column_index})
    if not labels:
      labels = [0]
    class_values.append(labels)
  return class_values


def _normalize_auxiliary_target_rows(
  targets: list[list[int]] | None,
  *,
  row_count: int,
  column_count: int,
) -> list[list[int]]:
  if column_count == 0:
    return [[] for _ in range(row_count)]
  if targets is None:
    raise ValueError("auxiliary_targets are required when auxiliary_target_columns are configured.")
  if len(targets) != row_count:
    raise ValueError("auxiliary_targets length must match the sequence count.")
  output: list[list[int]] = []
  for row in targets:
    if len(row) != column_count:
      raise ValueError("auxiliary target width does not match auxiliary_target_columns.")
    output.append([int(value) for value in row])
  return output


def _auxiliary_target_indices(targets: list[list[int]], class_values: list[list[int]]) -> list[list[int]]:
  indices: list[list[int]] = []
  lookup_tables = [
    {label: index for index, label in enumerate(labels)}
    for labels in class_values
  ]
  for row in targets:
    indices.append(
      [
        lookup_tables[column_index][int(label)]
        for column_index, label in enumerate(row)
      ]
    )
  return indices


def _auxiliary_classification_loss(
  logits: Any,
  targets: Any,
  *,
  target_columns: list[str],
  class_values: list[list[int]],
  loss_weights: dict[str, float],
  sample_weights: Any | None = None,
) -> Any:
  torch = _require_torch()
  offset = 0
  losses = []
  column_weights = []
  for column_index, labels in enumerate(class_values):
    width = len(labels)
    column_weight = _auxiliary_column_loss_weight(target_columns[column_index], loss_weights)
    column_logits = logits[:, offset : offset + width]
    offset += width
    if column_weight <= 0:
      continue
    column_targets = targets[:, column_index]
    losses.append(torch.nn.functional.cross_entropy(column_logits, column_targets, reduction="none"))
    column_weights.append(column_weight)
  if not losses:
    return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
  weights = torch.tensor(column_weights, dtype=logits.dtype, device=logits.device)
  stacked = torch.stack(losses, dim=-1)
  sample_loss = (stacked * weights).sum(dim=-1) / weights.sum().clamp_min(1e-12)
  return _weighted_mean(sample_loss, sample_weights)


def _auxiliary_column_loss_weight(target_column: str, loss_weights: dict[str, float]) -> float:
  value = loss_weights.get(target_column, 1.0)
  try:
    return max(0.0, float(value))
  except (TypeError, ValueError):
    return 1.0


def _weighted_mean(values: Any, sample_weights: Any | None) -> Any:
  if sample_weights is None:
    return values.mean()
  weights = sample_weights.to(dtype=values.dtype, device=values.device)
  denominator = weights.sum().clamp_min(1e-12)
  return (values * weights).sum() / denominator


def _normalize_sample_weights(sample_weights: list[float] | None) -> list[float]:
  if not sample_weights:
    return []
  values = [max(0.0, float(value)) for value in sample_weights]
  mean_value = sum(values) / len(values)
  if mean_value <= 0:
    return [1.0 for _ in values]
  return [value / mean_value for value in values]


def _auxiliary_output_row(
  *,
  logits: list[float],
  target_columns: list[str],
  class_values: list[list[int]],
) -> dict[str, float]:
  row: dict[str, float] = {}
  offset = 0
  for target_column, labels in zip(target_columns, class_values):
    width = len(labels)
    probabilities = _softmax(logits[offset : offset + width])
    offset += width
    if not probabilities:
      continue
    max_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    probability_by_label = {
      label: probabilities[index]
      for index, label in enumerate(labels)
    }
    sorted_probabilities = sorted(probabilities, reverse=True)
    top_probability = sorted_probabilities[0]
    runner_up_probability = sorted_probabilities[1] if len(sorted_probabilities) > 1 else 0.0
    row[_auxiliary_prediction_column(target_column)] = float(labels[max_index])
    for label in labels:
      row[_auxiliary_probability_column(target_column, label)] = probability_by_label[label]
    if -1 in probability_by_label and 1 in probability_by_label:
      row[_auxiliary_margin_column(target_column)] = probability_by_label[1] - probability_by_label[-1]
    row[_auxiliary_confidence_column(target_column)] = max(0.0, top_probability - runner_up_probability)
  return row


def _softmax(values: list[float]) -> list[float]:
  if not values:
    return []
  max_value = max(values)
  exp_values = [math.exp(value - max_value) for value in values]
  denominator = sum(exp_values)
  if denominator <= 0:
    return [1.0 / len(values) for _ in values]
  return [value / denominator for value in exp_values]


def _feature_stats(sequences: list[list[list[float]]]) -> tuple[list[float], list[float]]:
  if not sequences:
    return [], []
  feature_count = len(sequences[0][0])
  means: list[float] = []
  stds: list[float] = []
  for feature_index in range(feature_count):
    values = [
      sequence[step][feature_index]
      for sequence in sequences
      for step in range(len(sequence))
    ]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    means.append(mean)
    stds.append(max(math.sqrt(variance), 1e-6))
  return means, stds


def _target_stats(targets: list[list[float]]) -> tuple[list[float], list[float]]:
  if not targets:
    return [], []
  width = len(targets[0])
  means: list[float] = []
  stds: list[float] = []
  for column_index in range(width):
    values = [float(row[column_index]) for row in targets]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    means.append(mean)
    stds.append(max(math.sqrt(variance), 1e-6))
  return means, stds


def _normalize_sequences(
  sequences: list[list[list[float]]],
  means: list[float],
  stds: list[float],
) -> list[list[list[float]]]:
  return [
    [
      [
        (step[feature_index] - means[feature_index]) / stds[feature_index]
        for feature_index in range(len(step))
      ]
      for step in sequence
    ]
    for sequence in sequences
  ]


def _normalize_target_matrix(
  targets: list[list[float]],
  means: list[float],
  stds: list[float],
) -> list[list[float]]:
  return [
    [
      (float(row[column_index]) - means[column_index]) / stds[column_index]
      for column_index in range(len(row))
    ]
    for row in targets
  ]


def _split_index(length: int, validation_fraction: float) -> int:
  validation_rows = max(1, int(length * validation_fraction))
  return max(1, length - validation_rows)


def _prediction_quantile_column(target_column: str, quantile: float) -> str:
  return f"prediction_return_q{_quantile_label(quantile)}_{_target_suffix(target_column)}"


def _prediction_spread_column(target_column: str) -> str:
  return f"prediction_return_spread_{_target_suffix(target_column)}"


def _auxiliary_prediction_columns(target_column: str, class_values: list[int]) -> list[str]:
  columns = [_auxiliary_prediction_column(target_column)]
  if -1 in class_values and 1 in class_values:
    columns.append(_auxiliary_margin_column(target_column))
  columns.extend(_auxiliary_probability_column(target_column, label) for label in class_values)
  columns.append(_auxiliary_confidence_column(target_column))
  return columns


def _auxiliary_prediction_column(target_column: str) -> str:
  name, suffix = _auxiliary_name_and_suffix(target_column)
  return f"prediction_{name}_{suffix}"


def _auxiliary_margin_column(target_column: str) -> str:
  name, suffix = _auxiliary_name_and_suffix(target_column)
  return f"prediction_{name}_margin_{suffix}"


def _auxiliary_probability_column(target_column: str, label: int) -> str:
  _name, suffix = _auxiliary_name_and_suffix(target_column)
  return f"prediction_probability_{_auxiliary_label_name(target_column, label)}_{suffix}"


def _auxiliary_confidence_column(target_column: str) -> str:
  name, suffix = _auxiliary_name_and_suffix(target_column)
  return f"prediction_{name}_confidence_{suffix}"


def _auxiliary_name_and_suffix(target_column: str) -> tuple[str, str]:
  if target_column.startswith("future_trend_state_label_"):
    return "trend_state", target_column.removeprefix("future_trend_state_label_")
  if target_column.startswith("future_trend_break_label_"):
    return "trend_break", target_column.removeprefix("future_trend_break_label_")
  if target_column.startswith("future_trend_persistence_label_"):
    return "trend_persistence", target_column.removeprefix("future_trend_persistence_label_")
  if target_column.startswith("triple_barrier_label_"):
    return "triple_barrier", target_column.removeprefix("triple_barrier_label_")
  return "auxiliary", _label_slug(target_column)


def _auxiliary_label_name(target_column: str, label: int) -> str:
  if target_column.startswith("future_trend_state_label_"):
    return {-1: "downtrend", 0: "chop", 1: "uptrend"}.get(label, f"class_{_label_slug(str(label))}")
  if target_column.startswith("triple_barrier_label_"):
    return {-1: "stop", 0: "no_hit", 1: "profit"}.get(label, f"class_{_label_slug(str(label))}")
  if target_column.startswith("future_trend_break_label_"):
    return {0: "no_break", 1: "break"}.get(label, f"class_{_label_slug(str(label))}")
  if target_column.startswith("future_trend_persistence_label_"):
    return {0: "not_persistent", 1: "persistent"}.get(label, f"class_{_label_slug(str(label))}")
  return f"class_{_label_slug(str(label))}"


def _default_auxiliary_labels(target_column: str) -> list[int]:
  if target_column.startswith(("future_trend_state_label_", "triple_barrier_label_")):
    return [-1, 0, 1]
  if target_column.startswith(("future_trend_break_label_", "future_trend_persistence_label_")):
    return [0, 1]
  return [0, 1]


def _label_slug(value: str) -> str:
  return value.replace("-", "neg").replace(".", "_").replace(" ", "_")


def _target_suffix(target_column: str) -> str:
  if target_column.startswith("forward_return_"):
    return target_column.removeprefix("forward_return_")
  return target_column


def _quantile_label(quantile: float) -> str:
  return f"{int(round(quantile * 100)):02d}"


def _median_quantile(quantiles: tuple[float, ...]) -> float:
  if 0.5 in quantiles:
    return 0.5
  return quantiles[len(quantiles) // 2]


def _mae(targets: list[float], predictions: list[float]) -> float:
  return sum(abs(target - prediction) for target, prediction in zip(targets, predictions)) / len(targets)


def _rmse(targets: list[float], predictions: list[float]) -> float:
  return math.sqrt(sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions)) / len(targets))


def _directional_accuracy(targets: list[float], predictions: list[float]) -> float:
  return sum(
    1
    for target, prediction in zip(targets, predictions)
    if _sign(target) == _sign(prediction)
  ) / len(targets)


def _classification_accuracy(targets: list[int], predictions: list[int]) -> float:
  if not targets:
    return 0.0
  return sum(1 for target, prediction in zip(targets, predictions) if int(target) == int(prediction)) / len(targets)


def _classification_macro_recall(targets: list[int], predictions: list[int]) -> float:
  if not targets:
    return 0.0
  recalls: list[float] = []
  for label in sorted(set(targets)):
    label_predictions = [prediction for target, prediction in zip(targets, predictions) if target == label]
    if not label_predictions:
      continue
    recalls.append(sum(1 for prediction in label_predictions if prediction == label) / len(label_predictions))
  return sum(recalls) / len(recalls) if recalls else 0.0


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0


def _require_torch() -> Any:
  try:
    import torch
  except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModelDependencyError(
      "torch is required for hybrid transformer training. Install torch before using the hybrid model."
    ) from exc
  return torch
