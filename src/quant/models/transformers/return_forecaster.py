from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant.models.base import ModelDependencyError


@dataclass(frozen=True, slots=True)
class TransformerForecasterConfig:
  lookback: int = 32
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


class ReturnSequenceTransformer:
  model_name = "transformer_return_forecaster"
  task_type = "regression"

  def __init__(self, config: TransformerForecasterConfig | None = None) -> None:
    self.config = config or TransformerForecasterConfig()
    self.model: Any | None = None
    self.feature_names: list[str] = []
    self.feature_means: list[float] = []
    self.feature_stds: list[float] = []
    self.target_mean: float = 0.0
    self.target_std: float = 1.0

  def fit(
    self,
    sequences: list[list[list[float]]],
    targets: list[float],
    *,
    feature_names: list[str],
    validation_sequences: list[list[list[float]]] | None = None,
    validation_targets: list[float] | None = None,
  ) -> dict[str, float]:
    torch = _require_torch()
    if not sequences:
      raise ValueError("No sequences provided for transformer training.")

    self.feature_names = list(feature_names)
    if validation_sequences is None or validation_targets is None:
      split_index = _split_index(len(sequences), self.config.validation_fraction)
      train_sequences = sequences[:split_index]
      train_targets = targets[:split_index]
      valid_sequences = sequences[split_index:]
      valid_targets = targets[split_index:]
    else:
      train_sequences = sequences
      train_targets = targets
      valid_sequences = validation_sequences
      valid_targets = validation_targets

    self.feature_means, self.feature_stds = _feature_stats(train_sequences)
    self.target_mean, self.target_std = _target_stats(train_targets)
    normalized_train_sequences = _normalize_sequences(
      train_sequences,
      self.feature_means,
      self.feature_stds,
    )
    normalized_train_targets = _normalize_targets(
      train_targets,
      self.target_mean,
      self.target_std,
    )
    normalized_valid_sequences = _normalize_sequences(
      valid_sequences,
      self.feature_means,
      self.feature_stds,
    )
    normalized_valid_targets = _normalize_targets(
      valid_targets,
      self.target_mean,
      self.target_std,
    )

    device = torch.device(self.config.device)
    self.model = _SequenceRegressor(
      input_dim=len(self.feature_names),
      lookback=self.config.lookback,
      d_model=self.config.d_model,
      nhead=self.config.nhead,
      num_layers=self.config.num_layers,
      dim_feedforward=self.config.dim_feedforward,
      dropout=self.config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
      self.model.parameters(),
      lr=self.config.learning_rate,
      weight_decay=self.config.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    train_tensor = torch.tensor(normalized_train_sequences, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(normalized_train_targets, dtype=torch.float32, device=device).unsqueeze(1)

    best_validation = math.inf
    best_state = None

    for _epoch in range(self.config.epochs):
      self.model.train()
      batch_losses: list[float] = []
      for start in range(0, len(train_sequences), self.config.batch_size):
        end = start + self.config.batch_size
        batch_x = train_tensor[start:end]
        batch_y = target_tensor[start:end]
        optimizer.zero_grad()
        predictions = self.model(batch_x)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.item()))

      validation_loss = (
        self._evaluate(
          normalized_valid_sequences,
          normalized_valid_targets,
          loss_fn,
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

    train_predictions = self.predict(train_sequences)
    metrics = {
      "train_mae": _mae(train_predictions, train_targets),
      "train_rmse": _rmse(train_predictions, train_targets),
    }
    if valid_sequences:
      valid_predictions = self.predict(valid_sequences)
      metrics["validation_mae"] = _mae(valid_predictions, valid_targets)
      metrics["validation_rmse"] = _rmse(valid_predictions, valid_targets)
    return metrics

  def predict(self, sequences: list[list[list[float]]]) -> list[float]:
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
      predictions = self.model(tensor).squeeze(1).detach().cpu().tolist()
    return _denormalize_predictions(
      [float(value) for value in predictions],
      self.target_mean,
      self.target_std,
    )

  def save(self, path: Path) -> None:
    torch = _require_torch()
    if self.model is None:
      raise RuntimeError("Model has not been fit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "config": asdict(self.config),
        "feature_names": self.feature_names,
        "feature_means": self.feature_means,
        "feature_stds": self.feature_stds,
        "target_mean": self.target_mean,
        "target_std": self.target_std,
        "state_dict": self.model.state_dict(),
      },
      path,
    )

  @classmethod
  def load(cls, path: Path) -> "ReturnSequenceTransformer":
    torch = _require_torch()
    payload = torch.load(path, map_location="cpu")
    config = TransformerForecasterConfig(**payload["config"])
    instance = cls(config)
    instance.feature_names = list(payload["feature_names"])
    instance.feature_means = [float(value) for value in payload.get("feature_means", [])]
    instance.feature_stds = [float(value) for value in payload.get("feature_stds", [])]
    instance.target_mean = float(payload.get("target_mean", 0.0))
    instance.target_std = float(payload.get("target_std", 1.0))
    instance.model = _SequenceRegressor(
      input_dim=len(instance.feature_names),
      lookback=config.lookback,
      d_model=config.d_model,
      nhead=config.nhead,
      num_layers=config.num_layers,
      dim_feedforward=config.dim_feedforward,
      dropout=config.dropout,
    )
    instance.model.load_state_dict(payload["state_dict"])
    return instance

  def _evaluate(
    self,
    sequences: list[list[list[float]]],
    targets: list[float],
    loss_fn: Any,
    device: Any,
  ) -> float:
    torch = _require_torch()
    self.model.eval()
    with torch.no_grad():
      features = torch.tensor(sequences, dtype=torch.float32, device=device)
      labels = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)
      loss = loss_fn(self.model(features), labels)
    return float(loss.item())


class _SequenceRegressor:  # pragma: no cover - exercised only when torch is installed
  def __new__(
    cls,
    *,
    input_dim: int,
    lookback: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
  ) -> Any:
    torch = _require_torch()

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
          torch.nn.Linear(d_model, d_model // 2),
          torch.nn.GELU(),
          torch.nn.Linear(d_model // 2, 1),
        )

      def forward(self, inputs: Any) -> Any:
        hidden = self.input_projection(inputs)
        hidden = hidden + self.position_embedding.unsqueeze(0)
        encoded = self.encoder(hidden)
        pooled = encoded[:, -1, :]
        return self.head(pooled)

    return Module()


def _split_index(length: int, validation_fraction: float) -> int:
  if length < 2:
    return length
  validation_rows = max(1, int(length * validation_fraction))
  split_index = length - validation_rows
  return max(1, split_index)


def _mae(predictions: list[float], targets: list[float]) -> float:
  return sum(abs(pred - target) for pred, target in zip(predictions, targets)) / len(predictions)


def _rmse(predictions: list[float], targets: list[float]) -> float:
  return math.sqrt(sum((pred - target) ** 2 for pred, target in zip(predictions, targets)) / len(predictions))


def _require_torch() -> Any:
  try:
    import torch
  except ModuleNotFoundError as exc:
    raise ModelDependencyError(
      "PyTorch is required for transformer training. Install the 'deep' extra."
    ) from exc
  return torch


def _feature_stats(sequences: list[list[list[float]]]) -> tuple[list[float], list[float]]:
  if not sequences:
    return [], []

  feature_count = len(sequences[0][0])
  totals = [0.0] * feature_count
  total_squares = [0.0] * feature_count
  sample_count = 0

  for sequence in sequences:
    for timestep in sequence:
      sample_count += 1
      for index, value in enumerate(timestep):
        totals[index] += value
        total_squares[index] += value * value

  means = [total / sample_count for total in totals]
  variances = [
    max((total_squares[index] / sample_count) - (means[index] ** 2), 0.0)
    for index in range(feature_count)
  ]
  stds = [math.sqrt(variance) if variance > 0 else 1.0 for variance in variances]
  return means, stds


def _target_stats(targets: list[float]) -> tuple[float, float]:
  if not targets:
    return 0.0, 1.0
  mean = sum(targets) / len(targets)
  variance = sum((target - mean) ** 2 for target in targets) / len(targets)
  std = math.sqrt(variance) if variance > 0 else 1.0
  return mean, std


def _normalize_sequences(
  sequences: list[list[list[float]]],
  means: list[float],
  stds: list[float],
) -> list[list[list[float]]]:
  if not sequences:
    return []
  return [
    [
      [
        (value - means[index]) / stds[index]
        for index, value in enumerate(timestep)
      ]
      for timestep in sequence
    ]
    for sequence in sequences
  ]


def _normalize_targets(targets: list[float], mean: float, std: float) -> list[float]:
  return [(target - mean) / std for target in targets]


def _denormalize_predictions(predictions: list[float], mean: float, std: float) -> list[float]:
  return [(prediction * std) + mean for prediction in predictions]
