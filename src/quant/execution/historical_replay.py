from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant.execution.paper_trading import (
  PaperTradingConfig,
  append_ledger,
  append_order,
  append_signal,
  create_paper_ledger,
  create_paper_signal,
  load_model_from_artifact,
  load_optional_live_scoring_artifact,
  load_supporting_artifacts_for_live_scoring,
  resolve_artifact_dir,
  state_file_path,
  write_state,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HistoricalReplayConfig:
  artifact_dir: Path
  source_data_root: Path
  provider: str
  symbol: str
  feature_date: str
  artifact_search_root: Path | None = None
  replay_root: Path = Path("data/replay/demo")
  step_delay_seconds: float = 0.15
  start_row: int = 0
  max_rows: int | None = None
  control_poll_seconds: float = 0.1
  paper: PaperTradingConfig | None = None
  config_path: Path | None = None


@dataclass(slots=True)
class HistoricalReplaySummary:
  replay_root: Path
  feature_root: Path
  paper_root: Path
  runtime_root: Path
  rows_streamed: int = 0
  signals_written: int = 0
  orders_written: int = 0
  ledger_rows_written: int = 0
  feature_date: str = ""
  provider: str = ""
  symbol: str = ""
  started_at: str = ""
  completed_at: str = ""


def run_historical_replay(config: HistoricalReplayConfig) -> HistoricalReplaySummary:
  artifact_dir = resolve_artifact_dir(config.artifact_dir)
  metadata = _load_json(artifact_dir / "metadata.json")
  model = load_model_from_artifact(artifact_dir)
  supporting_artifacts = load_supporting_artifacts_for_live_scoring(
    artifact_dir=artifact_dir,
    metadata=metadata,
    artifact_search_root=config.artifact_search_root,
  )
  tradeability_artifact, tradeability_supporting_artifacts = load_optional_live_scoring_artifact(
    artifact_dir=(config.paper.tradeability_artifact_dir if config.paper is not None else None),
    artifact_search_root=config.artifact_search_root,
  )

  replay_root = config.replay_root
  feature_root = replay_root / "features"
  paper_root = replay_root / "paper"
  runtime_root = replay_root / "runtime"
  control_path = replay_control_path(replay_root)
  status_path = replay_status_path(replay_root)
  for path in (feature_root, paper_root, runtime_root):
    path.mkdir(parents=True, exist_ok=True)

  source_rows, fieldnames = load_source_rows(
    source_data_root=config.source_data_root,
    provider=config.provider,
    symbol=config.symbol,
    feature_date=config.feature_date,
  )
  replay_rows = source_rows[config.start_row :]
  if config.max_rows is not None:
    replay_rows = replay_rows[: config.max_rows]
  if not replay_rows:
    raise ValueError("Replay source slice is empty.")
  total_rows = len(replay_rows)
  preloaded_history_rows = source_rows[: config.start_row] if config.start_row > 0 else []

  output_path = replay_dataset_path(
    feature_root=feature_root,
    provider=config.provider,
    symbol=config.symbol,
    feature_date=config.feature_date,
  )
  _write_csv_header(output_path, fieldnames)

  effective_paper = config.paper or PaperTradingConfig(
    artifact_dir=artifact_dir,
    artifact_search_root=config.artifact_search_root,
    product=config.symbol,
    output_root=paper_root,
    take_profit_multiple=0.5,
  )
  effective_paper = _paper_config_for_replay(effective_paper, artifact_dir=artifact_dir, paper_root=paper_root)
  replay_state_path = state_file_path(paper_root, metadata)
  control = save_replay_control(
    control_path,
    step_delay_seconds=config.step_delay_seconds,
    paused=False,
  )

  summary = HistoricalReplaySummary(
    replay_root=replay_root,
    feature_root=feature_root,
    paper_root=paper_root,
    runtime_root=runtime_root,
    feature_date=config.feature_date,
    provider=config.provider,
    symbol=config.symbol,
    started_at=_format_dt(datetime.now(UTC)),
  )
  write_replay_status(
    status_path,
    summary=summary,
    total_rows=total_rows,
    latest_candle_start="",
    control=control,
    completed=False,
  )

  streamed_rows: list[dict[str, str]] = list(preloaded_history_rows)
  for index, row in enumerate(replay_rows, start=1):
    streamed_rows.append(row)
    _append_dataset_row(output_path, fieldnames, row)
    previous_state = _load_json(replay_state_path) if replay_state_path.exists() else {}
    generated_at = _parse_dt(row.get("candle_end") or row.get("candle_start") or _format_dt(datetime.now(UTC)))
    signal_payload, order_payload = create_paper_signal(
      row=row,
      feature_rows=streamed_rows,
      metadata=metadata,
      model=model,
      artifact_dir=artifact_dir,
      config=effective_paper,
      generated_at=generated_at,
      previous_position=float(previous_state.get("target_position", 0.0)),
      previous_state=previous_state,
      feature_path=output_path,
      runtime_root=runtime_root,
      supporting_artifacts=supporting_artifacts,
      tradeability_artifact=tradeability_artifact,
      tradeability_supporting_artifacts=tradeability_supporting_artifacts,
    )
    ledger_payload = create_paper_ledger(
      row=row,
      signal_payload=signal_payload,
      previous_state=previous_state,
    )

    append_signal(paper_root, signal_payload)
    if order_payload is not None:
      append_order(paper_root, order_payload)
      summary.orders_written += 1
    append_ledger(paper_root, ledger_payload)
    write_state(replay_state_path, signal_payload, ledger_payload)

    summary.rows_streamed += 1
    summary.signals_written += 1
    summary.ledger_rows_written += 1
    control = load_replay_control(control_path, default_delay=config.step_delay_seconds)
    write_replay_status(
      status_path,
      summary=summary,
      total_rows=total_rows,
      latest_candle_start=row.get("candle_start", ""),
      control=control,
      completed=False,
    )
    if index < total_rows:
      control = wait_for_replay_step(
        control_path=control_path,
        default_delay=config.step_delay_seconds,
        poll_seconds=config.control_poll_seconds,
      )

  summary.completed_at = _format_dt(datetime.now(UTC))
  control = load_replay_control(control_path, default_delay=config.step_delay_seconds)
  write_replay_status(
    status_path,
    summary=summary,
    total_rows=total_rows,
    latest_candle_start=replay_rows[-1].get("candle_start", ""),
    control=control,
    completed=True,
  )
  (replay_root / "replay_summary.json").write_text(
    json.dumps(
      {
        "replay_root": str(summary.replay_root),
        "feature_root": str(summary.feature_root),
        "paper_root": str(summary.paper_root),
        "runtime_root": str(summary.runtime_root),
        "rows_streamed": summary.rows_streamed,
        "signals_written": summary.signals_written,
        "orders_written": summary.orders_written,
        "ledger_rows_written": summary.ledger_rows_written,
        "feature_date": summary.feature_date,
        "provider": summary.provider,
        "symbol": summary.symbol,
        "started_at": summary.started_at,
        "completed_at": summary.completed_at,
        "artifact_dir": str(artifact_dir),
        "paper_config": _json_ready(asdict(effective_paper)),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return summary


def load_source_rows(
  *,
  source_data_root: Path,
  provider: str,
  symbol: str,
  feature_date: str,
) -> tuple[list[dict[str, str]], list[str]]:
  year, month, day = feature_date.split("-")
  path = (
    source_data_root
    / "training_datasets"
    / "market_supervised"
    / provider
    / symbol
    / year
    / month
    / f"{day}.csv"
  )
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    return list(reader), list(reader.fieldnames or [])


def replay_dataset_path(
  *,
  feature_root: Path,
  provider: str,
  symbol: str,
  feature_date: str,
) -> Path:
  year, month, day = feature_date.split("-")
  return (
    feature_root
    / "training_datasets"
    / "market_supervised"
    / provider
    / symbol
    / year
    / month
    / f"{day}.csv"
  )


def _write_csv_header(path: Path, fieldnames: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()


def _append_dataset_row(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
  with path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writerow({field: row.get(field, "") for field in fieldnames})


def _paper_config_for_replay(
  config: PaperTradingConfig,
  *,
  artifact_dir: Path,
  paper_root: Path,
) -> PaperTradingConfig:
  return PaperTradingConfig(
    artifact_dir=artifact_dir,
    product=config.product,
    artifact_search_root=config.artifact_search_root,
    tradeability_artifact_dir=config.tradeability_artifact_dir,
    raw_root=config.raw_root,
    staging_root=config.staging_root,
    feature_root=config.feature_root,
    output_root=paper_root,
    candle_interval=config.candle_interval,
    lookback_hours=config.lookback_hours,
    poll_interval_seconds=config.poll_interval_seconds,
    iterations=config.iterations,
    fee_bps=config.fee_bps,
    slippage_bps=config.slippage_bps,
    take_profit_multiple=config.take_profit_multiple,
    request_timeout_seconds=config.request_timeout_seconds,
    request_pause_seconds=config.request_pause_seconds,
    sizing=config.sizing,
    end_time=config.end_time,
    config_path=config.config_path,
  )


def replay_control_path(replay_root: Path) -> Path:
  return replay_root / "replay_control.json"


def replay_status_path(replay_root: Path) -> Path:
  return replay_root / "replay_status.json"


def load_replay_control(path: Path, *, default_delay: float) -> dict[str, Any]:
  if not path.exists():
    return save_replay_control(path, step_delay_seconds=default_delay, paused=False)
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return save_replay_control(path, step_delay_seconds=default_delay, paused=False)
  delay = _coerce_float(payload.get("step_delay_seconds"), default_delay)
  paused = bool(payload.get("paused", False))
  return {
    "step_delay_seconds": max(0.0, delay),
    "paused": paused,
    "updated_at": str(payload.get("updated_at") or ""),
  }


def save_replay_control(path: Path, *, step_delay_seconds: float, paused: bool) -> dict[str, Any]:
  payload = {
    "step_delay_seconds": max(0.0, float(step_delay_seconds)),
    "paused": bool(paused),
    "updated_at": _format_dt(datetime.now(UTC)),
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  return payload


def wait_for_replay_step(
  *,
  control_path: Path,
  default_delay: float,
  poll_seconds: float,
) -> dict[str, Any]:
  control = load_replay_control(control_path, default_delay=default_delay)
  while control.get("paused"):
    time.sleep(max(0.01, poll_seconds))
    control = load_replay_control(control_path, default_delay=default_delay)
  delay = _coerce_float(control.get("step_delay_seconds"), default_delay)
  if delay > 0:
    time.sleep(delay)
  return control


def write_replay_status(
  path: Path,
  *,
  summary: HistoricalReplaySummary,
  total_rows: int,
  latest_candle_start: str,
  control: dict[str, Any],
  completed: bool,
) -> None:
  payload = {
    "replay_root": str(summary.replay_root),
    "feature_root": str(summary.feature_root),
    "paper_root": str(summary.paper_root),
    "runtime_root": str(summary.runtime_root),
    "provider": summary.provider,
    "symbol": summary.symbol,
    "feature_date": summary.feature_date,
    "rows_streamed": summary.rows_streamed,
    "signals_written": summary.signals_written,
    "orders_written": summary.orders_written,
    "ledger_rows_written": summary.ledger_rows_written,
    "total_rows": total_rows,
    "progress_fraction": (summary.rows_streamed / total_rows) if total_rows else 0.0,
    "latest_candle_start": latest_candle_start,
    "step_delay_seconds": _coerce_float(control.get("step_delay_seconds"), 0.0),
    "paused": bool(control.get("paused", False)),
    "started_at": summary.started_at,
    "completed_at": summary.completed_at,
    "completed": completed,
    "updated_at": _format_dt(datetime.now(UTC)),
  }
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_dt(value: datetime) -> str:
  return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(key): _json_ready(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_ready(item) for item in value]
  return value


def _coerce_float(value: Any, default: float) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return float(default)
