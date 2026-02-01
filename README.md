# OsuManiaTransformer4K

Hybrid mapping project for **osu!mania 4K**.

Core idea: split the problem into:
- **The Ear**: timing / beat grid (Beat This + timing-point builder)
- **The Brain**: note/pattern generation (Transformer; roadmap)

Project anchor: see **The Four Foundations** in `roadmap.md`.

## Quick Start

### Inference (timing points)

```powershell
python Modules\beat_this\beat_to_osu.py path\to\audio.mp3 --timing-profile auto -o out.timing.txt
```

Shortcut: `.\tasks.ps1 ear:infer -Audio path\to\audio.mp3`

### Dataset gathering (build `Datasets2/`)

See `docs/DATASETS.md`.

### Training (Ear fine-tune → export ONNX back to Windows)

See `docs/TRAINING_SITE.md`.

## Repo Layout (where things live)

- `Modules/beat_this/`: Beat This model + `beat_to_osu.py` (timing inference, ONNX export)
- `Datasets/`: osu2mir dataset audio downloader + API tooling
- `Datasets2/`: `.osz` → 4K dataset builder + dataset utilities
- `scripts/`: dataset conversion + fine-tune scripts (osu-supervised Ear training)
- `training_site/`: WSL2 training runner + ROCm/WSL diagnostics helpers
- `osu2mir/`: upstream dataset tools + reference tables
- `docs/`: setup docs (dataset + training)

If VS Code explorer feels noisy, add the repo’s recommended excludes: `.vscode/settings.json` (optional).
