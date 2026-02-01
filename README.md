# OsuManiaTransformer4K

Hybrid mapping project for **osu!mania 4K**.

Core idea: split the problem into:
- **The Ear**: timing / beat grid (Beat This + timing-point builder)
- **The Brain**: note/pattern generation (Transformer; roadmap)

Project anchor: see **The Four Foundations** in `roadmap.md`.

## Quick Start

### Inference (timing points)

```powershell
python modules\beat_this\beat_to_osu.py path\to\audio.mp3 --timing-profile auto -o out.timing.txt
```

Shortcut: `.\tasks.ps1 ear:infer -Audio path\to\audio.mp3`

### Dataset gathering (build `data/datasets2/`)

See `docs/DATASETS.md`.

### Training (Ear fine-tune → export ONNX back to Windows)

See `docs/TRAINING_SITE.md`.

## Repo Layout (where things live)

- `modules/beat_this/`: Beat This model + `beat_to_osu.py` (timing inference, ONNX export)
- `data/osu2mir_audio/`: osu2mir audio downloader + API tooling
- `data/datasets2/`: `.osz` → 4K dataset builder + dataset utilities
- `training/scripts/`: dataset conversion + fine-tune scripts (osu-supervised Ear training)
- `training/wsl/`: WSL2 training runner + ROCm/WSL diagnostics helpers
- `data/osu2mir/`: upstream dataset tools + reference tables
- `docs/`: setup docs (dataset + training)

If VS Code explorer feels noisy, add the repo’s recommended excludes: `.vscode/settings.json` (optional).
