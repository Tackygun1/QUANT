from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ModelDependencyError(RuntimeError):
  """Raised when an optional ML dependency is missing."""


@dataclass(frozen=True, slots=True)
class TrainingArtifactMetadata:
  model_name: str
  task_type: str
  target_column: str
  provider: str
  symbol: str
  feature_columns: tuple[str, ...]
  train_rows: int
  validation_rows: int
  metrics: dict[str, float] = field(default_factory=dict)
  extra: dict[str, Any] = field(default_factory=dict)
  provenance: dict[str, Any] = field(default_factory=dict)

  def write_json(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(self)
    payload["provenance"] = {
      **build_training_provenance(payload, output_path=path),
      **self.provenance,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def utc_run_slug() -> str:
  return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def build_training_provenance(metadata_payload: dict[str, Any], *, output_path: Path) -> dict[str, Any]:
  config_path = metadata_payload.get("extra", {}).get("config_path") or ""
  artifact_checksums = {
    artifact_path.name: _sha256_file(artifact_path)
    for artifact_path in sorted(output_path.parent.iterdir())
    if artifact_path.is_file() and artifact_path != output_path
  }
  dependency_snapshot = _installed_dependency_snapshot()
  return {
    "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "git_commit": _git_output("rev-parse", "HEAD"),
    "git_dirty": bool(_git_output("status", "--porcelain")),
    "config_sha256": _sha256_file(Path(config_path)) if config_path and Path(config_path).exists() else "",
    "feature_schema_sha256": _sha256_json(metadata_payload.get("feature_columns", [])),
    "dependency_snapshot_sha256": _sha256_json(dependency_snapshot),
    "artifact_checksums": artifact_checksums,
  }


def verify_artifact_checksum(artifact_dir: Path, metadata: dict[str, Any], artifact_filename: str) -> None:
  expected = (
    metadata.get("provenance", {})
    .get("artifact_checksums", {})
    .get(artifact_filename)
  )
  if not expected:
    return
  artifact_path = artifact_dir / artifact_filename
  actual = _sha256_file(artifact_path)
  if actual != expected:
    raise RuntimeError(
      f"Artifact checksum mismatch for {artifact_path}: expected {expected}, got {actual}."
    )


def _installed_dependency_snapshot() -> list[dict[str, str]]:
  return sorted(
    (
      {"name": distribution.metadata["Name"].lower(), "version": distribution.version}
      for distribution in importlib.metadata.distributions()
      if distribution.metadata.get("Name")
    ),
    key=lambda item: item["name"],
  )


def _sha256_json(value: Any) -> str:
  encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_output(*args: str) -> str:
  try:
    result = subprocess.run(
      ("git", *args),
      check=False,
      cwd=Path(__file__).resolve().parents[2],
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
      timeout=5,
    )
  except (OSError, subprocess.TimeoutExpired):
    return ""
  if result.returncode != 0:
    return ""
  return result.stdout.strip()
