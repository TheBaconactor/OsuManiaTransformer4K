param(
  [string]$WslRepo = "~/repos/OsuManiaTransformer4K-train",
  [string]$OnnxName = "osu_ear_finetune",
  [string]$ValArtist = "Rousseau",
  [int]$Epochs = 10,
  [int]$BatchSize = 4,
  [int]$TrainLength = 1500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Convert-ToWslPath([string]$WindowsPath) {
  $p = $WindowsPath.Replace('\', '/')
  if ($p -match '^([A-Za-z]):/(.*)$') {
    $drive = $matches[1].ToLower()
    $rest = $matches[2]
    return "/mnt/$drive/$rest"
  }
  throw "Unsupported path format: $WindowsPath"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRootWsl = Convert-ToWslPath $RepoRoot
$OnnxWin = Join-Path $RepoRoot "Modules\beat_this\beat_this\onnx_models\$OnnxName.onnx"
$OnnxWinWsl = Convert-ToWslPath $OnnxWin

Write-Host "[1/5] Sync repo to WSL ($WslRepo)"
wsl bash -lc "set -euo pipefail; mkdir -p $WslRepo; rsync -a --delete --exclude '.venv/' --exclude '__pycache__/' --exclude 'reports/' --exclude 'Modules/beat_this/data/' --exclude 'Modules/beat_this/checkpoints/' '$RepoRootWsl/' '$WslRepo/'"

Write-Host "[2/5] Install system deps in WSL (ffmpeg, python3-venv)"
wsl -u root bash -lc "set -euo pipefail; if ls /etc/apt/sources.list.d/*nvidia*cuda* >/dev/null 2>&1; then for f in /etc/apt/sources.list.d/*nvidia*cuda*; do mv -f \"\$f\" \"\$f.disabled\"; done; fi; apt-get update -y; apt-get install -y ffmpeg git python3-venv build-essential rocminfo rocm-smi >/dev/null"

Write-Host "[3/5] Create venv + install Python deps in WSL"
wsl bash -lc "set -euo pipefail; cd $WslRepo; python3 -m venv .venv; source .venv/bin/activate; python -m pip install -U pip >/dev/null; pip install -r training_site/requirements-train.txt >/dev/null; pip install -e Modules/beat_this >/dev/null; if rocminfo 2>/dev/null | head -n 2 | grep -qi 'ROCk module is NOT loaded'; then TORCH_INDEX='cpu'; else TORCH_INDEX='rocm6.2'; fi; pip install --index-url https://download.pytorch.org/whl/\$TORCH_INDEX torch torchaudio >/dev/null"

Write-Host "[4/5] Run Ear fine-tune + export ONNX (name=$OnnxName)"
wsl bash -lc "set -euo pipefail; cd $WslRepo; source .venv/bin/activate; ONNX_NAME='$OnnxName' VAL_ARTIST='$ValArtist' EPOCHS='$Epochs' BATCH_SIZE='$BatchSize' TRAIN_LENGTH='$TrainLength' GPU='auto' ./training_site/run_ear_finetune.sh"

Write-Host "[5/5] Copy ONNX back to Windows ($OnnxWin)"
wsl bash -lc "set -euo pipefail; cp -f '$WslRepo/Modules/beat_this/beat_this/onnx_models/$OnnxName.onnx' '$OnnxWinWsl'"

Write-Host "[OK] Done. Inference usage:"
Write-Host "  python Modules/beat_this/beat_to_osu.py <audio> --device dml --checkpoint $OnnxName --timing-profile live -o out.timing.txt"

