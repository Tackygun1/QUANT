from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from quant.models.base import TrainingArtifactMetadata, utc_run_slug
from quant.models.transformers.return_forecaster import (
  ReturnSequenceTransformer,
  TransformerForecasterConfig,
)
from quant.training.datasets import (
  build_sequence_dataset,
  chronological_split,
  discover_series,
  load_series_rows,
  subset_sequence_dataset,
)


@dataclass(frozen=True, slots=True)
class TransformerTrainingConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/models")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  target_column: str = "forward_return_1_bar"
  minimum_rows: int = 64
  config_path: Path | None = None
  transformer: TransformerForecasterConfig = TransformerForecasterConfig()


@dataclass(slots=True)
class TransformerTrainingSummary:
  runs_attempted: int = 0
  runs_completed: int = 0
  artifacts_written: int = 0


def train_transformer_forecasters(config: TransformerTrainingConfig) -> TransformerTrainingSummary:
  summary = TransformerTrainingSummary()

  for series_key in discover_series(config.input_root, providers=config.providers, symbols=config.symbols):
    summary.runs_attempted += 1
    rows = load_series_rows(config.input_root, series_key)
    dataset = build_sequence_dataset(
      rows,
      target_column=config.target_column,
      lookback=config.transformer.lookback,
    )
    if len(dataset.rows) < config.minimum_rows:
      continue

    train_indices, validation_indices = chronological_split(
      len(dataset.rows),
      config.transformer.validation_fraction,
    )
    train_dataset = subset_sequence_dataset(dataset, train_indices)
    validation_dataset = subset_sequence_dataset(dataset, validation_indices)

    model = ReturnSequenceTransformer(config.transformer)
    metrics = model.fit(
      train_dataset.sequences,
      train_dataset.targets,
      feature_names=train_dataset.feature_names,
      validation_sequences=validation_dataset.sequences,
      validation_targets=validation_dataset.targets,
    )

    artifact_root = (
      config.output_root
      / "transformers"
      / "return_forecaster"
      / series_key.provider
      / series_key.symbol
      / utc_run_slug()
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    model_path = artifact_root / "model.pt"
    metadata_path = artifact_root / "metadata.json"
    predictions_path = artifact_root / "validation_predictions.csv"

    model.save(model_path)
    TrainingArtifactMetadata(
      model_name=model.model_name,
      task_type=model.task_type,
      target_column=config.target_column,
      provider=series_key.provider,
      symbol=series_key.symbol,
      feature_columns=tuple(dataset.feature_names),
      train_rows=len(train_dataset.rows),
      validation_rows=len(validation_dataset.rows),
      metrics=metrics,
      extra={
        "config_path": str(config.config_path) if config.config_path else "",
        "transformer_config": asdict(config.transformer),
      },
    ).write_json(metadata_path)

    validation_predictions = model.predict(validation_dataset.sequences) if validation_dataset.sequences else []
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(
        handle,
        fieldnames=["provider", "symbol", "candle_start", "target", "prediction"],
      )
      writer.writeheader()
      for row, target, prediction in zip(
        validation_dataset.rows,
        validation_dataset.targets,
        validation_predictions,
      ):
        writer.writerow(
          {
            "provider": row.get("provider", ""),
            "symbol": row.get("symbol", ""),
            "candle_start": row.get("candle_start", ""),
            "target": target,
            "prediction": prediction,
          }
        )

    (artifact_root / "training_config.json").write_text(
      json.dumps(
        {
          "input_root": str(config.input_root),
          "target_column": config.target_column,
          "minimum_rows": config.minimum_rows,
          "transformer": asdict(config.transformer),
        },
        indent=2,
        sort_keys=True,
      ),
      encoding="utf-8",
    )

    summary.runs_completed += 1
    summary.artifacts_written += 3

  return summary
