# Training Site (WSL2 / AMD GPU) + Export Back to Windows

Goal: do **training/fine-tuning** in a Linux environment (recommended for AMD GPUs), then **export artifacts back** into this Windows repo for inference.

This keeps the Windows repo focused on inference + tooling, while training happens elsewhere.

## Quick start (from Windows)

Run the one-shot driver script (sync → setup → train → export → copy back):

```powershell
powershell -ExecutionPolicy Bypass -File training_site\run_wsl_ear_finetune.ps1
```

Tuning knobs:
- `-ValArtist` (default: `Rousseau`) chooses which artist is held out as validation.
- `-Epochs`, `-BatchSize`, `-TrainLength` control training speed/quality.
- Output ONNX name: `-OnnxName osu_ear_finetune` (default).

## Dataset gathering

See `docs/DATASETS.md` for the supported dataset flows (osu! exports vs API search + mirror download) and how to build `Datasets2/`.

## Recommended layout

- **Inference repo (Windows):** `C:\Users\troll\Desktop\OsuManiaTransformer4K`
- **Training repo (WSL2 ext4):** `~/repos/OsuManiaTransformer4K-train`

Why: training inside `/home/...` is much faster than training on `/mnt/c/...`.

## One-time setup (training repo)

If you prefer manual setup (instead of the PowerShell script), do:

1) Sync the repo into WSL2 (preserves local/uncommitted changes better than `git clone`):
```bash
mkdir -p ~/repos/OsuManiaTransformer4K-train
rsync -a --delete /mnt/c/Users/troll/Desktop/OsuManiaTransformer4K/ ~/repos/OsuManiaTransformer4K-train/
cd ~/repos/OsuManiaTransformer4K-train
```

2) Install system deps (run as root if your WSL user needs a sudo password):
```bash
apt-get update -y
apt-get install -y ffmpeg git python3-venv build-essential rocminfo rocm-smi
```

3) Create a venv and install deps.

- Install a PyTorch build that supports your AMD GPU (ROCm) for WSL2, then:
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r training_site/requirements-train.txt
pip install -e Modules/beat_this

# Choose ONE:
# - ROCm (WSL2 + AMD GPU): huge download, only works if ROCm kernel driver is available
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchaudio
# - CPU fallback:
# pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
```

### ROCm sanity check (WSL2)

If ROCm is working, these should *not* error:
```bash
rocminfo | head
rocm-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

If you see `ROCk module is NOT loaded` / `amdgpu not found in modules`, then ROCm GPU training is **not available** in this WSL kernel/driver setup and training will run on CPU.

### Troubleshooting

- `apt-get update` fails with an NVIDIA CUDA repo signature error:
  - Disable the offending file under `/etc/apt/sources.list.d/` (our PowerShell runner does this automatically).
- Quick diagnostic script (run inside WSL2):
  - `./training_site/diagnose_rocm_wsl.sh`

## Fine-tune “The Ear” (osu-supervised, no human feedback)

From the training repo:
```bash
./training_site/run_ear_finetune.sh
```

What it does:
- builds a Beat This-style dataset from `Datasets2/`
- fine-tunes Beat This on those beat labels
- exports an ONNX model for DirectML inference

## Export back to Windows (for inference)

The script writes:
- `Modules/beat_this/beat_this/onnx_models/osu_ear_finetune.onnx`

Copy it to the Windows inference repo:
```bash
cp Modules/beat_this/beat_this/onnx_models/osu_ear_finetune.onnx \
  /mnt/c/Users/troll/Desktop/OsuManiaTransformer4K/Modules/beat_this/beat_this/onnx_models/
```

## Using the exported model on Windows (DirectML)

From the Windows inference repo:
```powershell
python Modules/beat_this/beat_to_osu.py Datasets2/audio/-1_0_terceS_Hard.mp3 `
  --device dml `
  --checkpoint osu_ear_finetune `
  --timing-profile live `
  -o out.timing.txt
```

Notes:
- `--checkpoint osu_ear_finetune` maps to `Modules/beat_this/beat_this/onnx_models/osu_ear_finetune.onnx`.
- If you don’t use `--device dml`, PyTorch will be used (CPU unless you have CUDA).
