#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # Prefer repo-local venv if present.
  source "$ROOT/.venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

DATASET_NAME="${DATASET_NAME:-osu_datasets2}"
VAL_ARTIST="${VAL_ARTIST:-Rousseau}"
TUNE="${TUNE:-heads}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_LENGTH="${TRAIN_LENGTH:-1500}"
GPU="${GPU:-auto}"

ONNX_NAME="${ONNX_NAME:-osu_ear_finetune}"
ONNX_OUT="modules/beat_this/beat_this/onnx_models/${ONNX_NAME}.onnx"

if [[ "${GPU}" == "auto" ]]; then
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
ok = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
raise SystemExit(0 if ok else 1)
PY
  then
    GPU="0"
  else
    GPU="-1"
    echo "[WARN] No CUDA/HIP device detected; training will run on CPU."
    if command -v rocminfo >/dev/null 2>&1; then
      if rocminfo 2>/dev/null | head -n 1 | grep -qi "ROCK module is NOT loaded"; then
        echo "[WARN] rocminfo reports 'ROCk module is NOT loaded' (ROCm GPU training not available in this WSL kernel/driver setup)."
      fi
    fi
  fi
fi

echo "[1/3] Build Beat This dataset from data/datasets2 (dataset=${DATASET_NAME}, val_artist=${VAL_ARTIST})"
"${PYTHON_BIN}" training/scripts/build_beat_this_osu_dataset.py \
  --dataset-name "${DATASET_NAME}" \
  --val-artist "${VAL_ARTIST}" \
  --datasets2 "data/datasets2" \
  --spect-dtype float32

echo "[2/3] Fine-tune Beat This (tune=${TUNE}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, gpu=${GPU})"
"${PYTHON_BIN}" training/scripts/finetune_beat_this_osu.py \
  --init-checkpoint final0 \
  --tune "${TUNE}" \
  --max-epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --train-length "${TRAIN_LENGTH}" \
  --gpu "${GPU}"

echo "[3/3] Export ONNX for Windows DirectML inference (${ONNX_OUT})"
"${PYTHON_BIN}" modules/beat_this/export_beat_this_onnx.py \
  --checkpoint "modules/beat_this/checkpoints/osu_finetune/last.ckpt" \
  --output "${ONNX_OUT}"

echo "[OK] Exported: ${ONNX_OUT}"
echo "Copy it to Windows inference repo on C: and run:"
echo "  python modules/beat_this/beat_to_osu.py <audio> --device dml --checkpoint ${ONNX_NAME} --timing-profile live -o out.timing.txt"
