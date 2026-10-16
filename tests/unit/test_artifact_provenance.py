from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant.models.base import TrainingArtifactMetadata, verify_artifact_checksum


class ArtifactProvenanceTests(unittest.TestCase):
  def test_metadata_records_and_verifies_artifact_checksum(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      artifact_dir = Path(tmp_dir)
      model_path = artifact_dir / "model.pkl"
      model_path.write_bytes(b"model-bytes")
      metadata_path = artifact_dir / "metadata.json"

      TrainingArtifactMetadata(
        model_name="ridge_return",
        task_type="regression",
        target_column="forward_return_1_bar",
        provider="coinbase",
        symbol="BTC-USD",
        feature_columns=("close", "volume"),
        train_rows=10,
        validation_rows=4,
      ).write_json(metadata_path)

      payload = json.loads(metadata_path.read_text(encoding="utf-8"))
      self.assertIn("provenance", payload)
      self.assertIn("feature_schema_sha256", payload["provenance"])
      self.assertEqual(
        payload["provenance"]["artifact_checksums"]["model.pkl"],
        "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
      )

      verify_artifact_checksum(artifact_dir, payload, "model.pkl")
      model_path.write_bytes(b"changed")
      with self.assertRaises(RuntimeError):
        verify_artifact_checksum(artifact_dir, payload, "model.pkl")


if __name__ == "__main__":
  unittest.main()
