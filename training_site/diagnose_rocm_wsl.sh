#!/usr/bin/env bash
set -euo pipefail

echo "== WSL kernel =="
uname -a || true

echo "== Device nodes =="
ls -l /dev/dxg 2>/dev/null || echo "/dev/dxg: missing"
ls -l /dev/kfd 2>/dev/null || echo "/dev/kfd: missing"
ls -l /dev/dri 2>/dev/null || echo "/dev/dri: missing"
ls -l /dev/dri/renderD* 2>/dev/null || true

echo "== Kernel modules (amdgpu/kfd/dxg) =="
lsmod | grep -E 'amdgpu|kfd|dxg' || echo "No amdgpu/kfd modules loaded"

echo "== rocminfo =="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | head -n 20 || true
else
  echo "rocminfo not installed"
fi

echo "== rocm-smi =="
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || true
else
  echo "rocm-smi not installed"
fi

echo "== PyTorch =="
python3 - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
    print("hip", getattr(torch.version, "hip", None))
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
except Exception as e:
    print("torch import failed:", e)
PY

