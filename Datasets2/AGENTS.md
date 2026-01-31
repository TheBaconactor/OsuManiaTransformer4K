# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `Datasets2/` and below.
- **Key scripts:**
  - osz_to_dataset.py (osu! exports -> audio + annotations)
  - dataset_loader.py (helpers for loading data)

## Commands
- **Process all exports:** `python Datasets2/osz_to_dataset.py`
- **Process single file:** `python Datasets2/osz_to_dataset.py --file <beatmap.osz>`

## Conventions
- Inputs come from `C:\Users\troll\AppData\Local\osu!\Exports` (Windows osu! exports).
- Outputs go to `Datasets2/audio`, `Datasets2/annotations`, `Datasets2/timing_points` (gitignored).
- Only 4K mania maps are kept; others are skipped.

## Do not
- Commit generated audio/annotation data.
