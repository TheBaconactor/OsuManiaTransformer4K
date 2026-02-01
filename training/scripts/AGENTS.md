# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `training/scripts/` and below.
- **Key scripts:**
  - build_beat_this_osu_dataset.py (`data/datasets2` -> Beat This data)
  - finetune_beat_this_osu.py (fine-tune Beat This on osu-derived beats)

## Commands
- **Build dataset:** `python training/scripts/build_beat_this_osu_dataset.py --val-artist Rousseau`
- **Smoke test:** `python training/scripts/build_beat_this_osu_dataset.py --max-items 1 --no-spectrograms`
- **Fine-tune:** `python training/scripts/finetune_beat_this_osu.py --init-checkpoint final0 --tune heads`

## Conventions
- Requires `data/datasets2/audio` + `data/datasets2/annotations` to exist.
- Outputs go under `modules/beat_this/data` and `modules/beat_this/checkpoints` (both gitignored).
- Keep these scripts runnable from repo root (they patch sys.path for `modules/beat_this`).

## Do not
- Commit generated datasets, checkpoints, or ONNX exports.
