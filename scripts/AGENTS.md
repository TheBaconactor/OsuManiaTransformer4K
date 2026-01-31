# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `scripts/` and below.
- **Key scripts:**
  - build_beat_this_osu_dataset.py (Datasets2 -> Beat This data)
  - finetune_beat_this_osu.py (fine-tune Beat This on osu-derived beats)

## Commands
- **Build dataset:** `python scripts/build_beat_this_osu_dataset.py --val-artist Rousseau`
- **Smoke test:** `python scripts/build_beat_this_osu_dataset.py --max-items 1 --no-spectrograms`
- **Fine-tune:** `python scripts/finetune_beat_this_osu.py --init-checkpoint final0 --tune heads`

## Conventions
- Requires `Datasets2/audio` + `Datasets2/annotations` to exist.
- Outputs go under `Modules/beat_this/data` and `Modules/beat_this/checkpoints` (both gitignored).
- Keep these scripts runnable from repo root (they patch sys.path for `Modules/beat_this`).

## Do not
- Commit generated datasets, checkpoints, or ONNX exports.
