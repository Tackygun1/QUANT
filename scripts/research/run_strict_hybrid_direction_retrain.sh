#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-experiments/retrains/coinbase/BTC-USD/20260418T200205Z}"
PROVIDER="${2:-coinbase}"
SYMBOL="${3:-BTC-USD}"
RUN_SLUG="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${WORKSPACE_ROOT}/strict_hybrid_oof/${RUN_SLUG}"
FEATURE_ROOT="${WORKSPACE_ROOT}/features"
STRICT_MODEL_ROOT="${RUN_DIR}/models"
LOG_DIR="${RUN_DIR}/logs"
STATE_PATH="${RUN_DIR}/state/strict_retrain_state.json"
STACKED_CONFIG_PATH="${RUN_DIR}/configs/stacked_dataset.json"
META_CONFIG_PATH="${RUN_DIR}/configs/meta_models.json"
SUMMARY_PATH="${RUN_DIR}/comparison_summary.json"
STACKED_DATASET_NAME="market_hybrid_stacked_strict_${RUN_SLUG}"
CURRENT_STEP="initializing"

mkdir -p "${RUN_DIR}/state" "${RUN_DIR}/configs" "${LOG_DIR}"

update_state() {
  local status="$1"
  local step="$2"
  local detail="${3:-}"
  STATE_PATH_ENV="${STATE_PATH}" \
  RUN_DIR_ENV="${RUN_DIR}" \
  FEATURE_ROOT_ENV="${FEATURE_ROOT}" \
  STRICT_MODEL_ROOT_ENV="${STRICT_MODEL_ROOT}" \
  STACKED_DATASET_NAME_ENV="${STACKED_DATASET_NAME}" \
  STATUS_ENV="${status}" \
  STEP_ENV="${step}" \
  DETAIL_ENV="${detail}" \
  ./.venv/bin/python - <<'PY'
import json
import os
from datetime import datetime, UTC
from pathlib import Path

state_path = Path(os.environ["STATE_PATH_ENV"])
state_path.parent.mkdir(parents=True, exist_ok=True)
payload = {}
if state_path.exists():
    payload = json.loads(state_path.read_text(encoding="utf-8"))
payload.update(
    {
        "status": os.environ["STATUS_ENV"],
        "current_step": os.environ["STEP_ENV"],
        "detail": os.environ.get("DETAIL_ENV", ""),
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_dir": os.environ["RUN_DIR_ENV"],
        "feature_root": os.environ["FEATURE_ROOT_ENV"],
        "strict_model_root": os.environ["STRICT_MODEL_ROOT_ENV"],
        "stacked_dataset_name": os.environ["STACKED_DATASET_NAME_ENV"],
    }
)
state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

on_exit() {
  local exit_code="$1"
  if [[ "${exit_code}" -ne 0 ]]; then
    update_state "failed" "${CURRENT_STEP}" "exit_code=${exit_code}"
  fi
}

trap 'on_exit $?' EXIT

update_state "running" "${CURRENT_STEP}" ""

CURRENT_STEP="walk_forward"
update_state "running" "${CURRENT_STEP}" "starting strict purged hybrid walk-forward"
./.venv/bin/python scripts/models/run_hybrid_walk_forward_validation.py \
  --input-root "${FEATURE_ROOT}" \
  --output-root "${STRICT_MODEL_ROOT}" \
  --provider "${PROVIDER}" \
  --symbol "${SYMBOL}" \
  --dataset-name market_supervised \
  --target-column forward_return_1_bar \
  --target-column forward_return_6_bar \
  --target-column forward_return_24_bar \
  --target-column forward_return_96_bar \
  --initial-train-rows 2048 \
  --validation-rows 512 \
  --step-rows 512 \
  --embargo-rows 1 \
  --purge-rows 96 \
  --minimum-rows 2048 \
  --backbone timemixer \
  --scale 1 \
  --scale 2 \
  --scale 4 \
  --mixer-layers 2 \
  --lookback 48 \
  --d-model 64 \
  --nhead 4 \
  --num-layers 2 \
  --dim-feedforward 128 \
  --batch-size 32 \
  --epochs 16 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --device cpu

CURRENT_STEP="stacked_dataset"
update_state "running" "${CURRENT_STEP}" "building strict hybrid stacked dataset from OOF predictions"
STACKED_CONFIG_PATH_ENV="${STACKED_CONFIG_PATH}" \
FEATURE_ROOT_ENV="${FEATURE_ROOT}" \
STRICT_MODEL_ROOT_ENV="${STRICT_MODEL_ROOT}" \
PROVIDER_ENV="${PROVIDER}" \
SYMBOL_ENV="${SYMBOL}" \
STACKED_DATASET_NAME_ENV="${STACKED_DATASET_NAME}" \
./.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "input_root": os.environ["FEATURE_ROOT_ENV"],
    "prediction_root": str(Path(os.environ["STRICT_MODEL_ROOT_ENV"]) / "walk_forward"),
    "output_root": os.environ["FEATURE_ROOT_ENV"],
    "input_dataset_name": "market_supervised",
    "output_dataset_name": os.environ["STACKED_DATASET_NAME_ENV"],
    "providers": [os.environ["PROVIDER_ENV"]],
    "symbols": [os.environ["SYMBOL_ENV"]],
    "prediction_sources": [
        {
            "family": "hybrid_transformers",
            "model_name": "quantile_forecaster",
            "alias": "hybrid_transformer",
            "source_columns": [
                "prediction_return_q10_1_bar",
                "prediction_return_q50_1_bar",
                "prediction_return_q90_1_bar",
                "prediction_return_spread_1_bar",
                "prediction_return_q10_6_bar",
                "prediction_return_q50_6_bar",
                "prediction_return_q90_6_bar",
                "prediction_return_spread_6_bar",
                "prediction_return_q10_24_bar",
                "prediction_return_q50_24_bar",
                "prediction_return_q90_24_bar",
                "prediction_return_spread_24_bar",
                "prediction_return_q10_96_bar",
                "prediction_return_q50_96_bar",
                "prediction_return_q90_96_bar",
                "prediction_return_spread_96_bar",
            ],
        }
    ],
}
path = Path(os.environ["STACKED_CONFIG_PATH_ENV"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

./.venv/bin/python scripts/models/build_stacked_dataset.py --config "${STACKED_CONFIG_PATH}"

CURRENT_STEP="meta_train"
update_state "running" "${CURRENT_STEP}" "training strict regime-aware hybrid direction meta-model"
META_CONFIG_PATH_ENV="${META_CONFIG_PATH}" \
FEATURE_ROOT_ENV="${FEATURE_ROOT}" \
STRICT_MODEL_ROOT_ENV="${STRICT_MODEL_ROOT}" \
PROVIDER_ENV="${PROVIDER}" \
SYMBOL_ENV="${SYMBOL}" \
STACKED_DATASET_NAME_ENV="${STACKED_DATASET_NAME}" \
./.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "input_root": os.environ["FEATURE_ROOT_ENV"],
    "output_root": os.environ["STRICT_MODEL_ROOT_ENV"],
    "dataset_name": os.environ["STACKED_DATASET_NAME_ENV"],
    "providers": [os.environ["PROVIDER_ENV"]],
    "symbols": [os.environ["SYMBOL_ENV"]],
    "model_names": ["lightgbm_hybrid_direction_meta"],
    "feature_groups": ["baseline_core", "layered_context", "hybrid_transformer_oof"],
    "validation_fraction": 0.2,
    "minimum_rows": 512,
    "sample_weight_scheme": "hmm_regime_target_balance",
    "regime_change_threshold": 0.35,
    "regime_change_boost": 1.5,
}
path = Path(os.environ["META_CONFIG_PATH_ENV"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

./.venv/bin/python scripts/models/train_meta_models.py --config "${META_CONFIG_PATH}"

CURRENT_STEP="compare"
update_state "running" "${CURRENT_STEP}" "writing strict-vs-fast retrain comparison summary"
WORKSPACE_ROOT_ENV="${WORKSPACE_ROOT}" \
STRICT_MODEL_ROOT_ENV="${STRICT_MODEL_ROOT}" \
PROVIDER_ENV="${PROVIDER}" \
SYMBOL_ENV="${SYMBOL}" \
SUMMARY_PATH_ENV="${SUMMARY_PATH}" \
STACKED_DATASET_NAME_ENV="${STACKED_DATASET_NAME}" \
./.venv/bin/python - <<'PY'
import csv
import json
import os
from pathlib import Path

workspace_root = Path(os.environ["WORKSPACE_ROOT_ENV"])
strict_model_root = Path(os.environ["STRICT_MODEL_ROOT_ENV"])
provider = os.environ["PROVIDER_ENV"]
symbol = os.environ["SYMBOL_ENV"]
summary_path = Path(os.environ["SUMMARY_PATH_ENV"])
stacked_dataset_name = os.environ["STACKED_DATASET_NAME_ENV"]

fast_root = workspace_root / "models" / "ensembles" / "lightgbm_hybrid_direction_meta" / provider / symbol
strict_root = strict_model_root / "ensembles" / "lightgbm_hybrid_direction_meta" / provider / symbol
walk_root = strict_model_root / "walk_forward" / "hybrid_transformers" / "quantile_forecaster" / provider / symbol
stacked_root = workspace_root / "features" / "training_datasets" / stacked_dataset_name / provider / symbol

def latest_run(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(root)
    return candidates[-1]

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def dataset_stats(root: Path) -> dict:
    rows = 0
    files = sorted(root.rglob("*.csv"))
    first = None
    last = None
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                candle_start = row.get("candle_start", "")
                if first is None or candle_start < first:
                    first = candle_start
                if last is None or candle_start > last:
                    last = candle_start
    return {
        "files": len(files),
        "rows": rows,
        "first_candle_start": first or "",
        "last_candle_start": last or "",
    }

fast_artifact = latest_run(fast_root)
strict_artifact = latest_run(strict_root)
walk_artifact = latest_run(walk_root)

payload = {
    "fast_artifact_dir": str(fast_artifact),
    "strict_artifact_dir": str(strict_artifact),
    "strict_walk_forward_dir": str(walk_artifact),
    "stacked_dataset_name": stacked_dataset_name,
    "stacked_dataset_stats": dataset_stats(stacked_root),
    "fast_metrics": load_json(fast_artifact / "metadata.json").get("metrics", {}),
    "strict_metrics": load_json(strict_artifact / "metadata.json").get("metrics", {}),
    "strict_walk_forward_summary": load_json(walk_artifact / "summary.json"),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

update_state "completed" "completed" "strict hybrid OOF retrain finished"
trap - EXIT
