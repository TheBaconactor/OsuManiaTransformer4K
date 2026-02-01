# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `data/osu2mir/` and below.
- **Key scripts:**
  - data_partition.py (partition .osz files)
  - data_conversion.py (convert .osz to audio + metered beats)
  - download_beatmaps.py (osu! beatmap downloader)
  - additional_tools/ (analysis/evaluation helpers)

## Commands
- **Partition data:** `python data/osu2mir/data_partition.py`
- **Convert to metered beats:** `python data/osu2mir/data_conversion.py`
- **Download beatmaps:** `python data/osu2mir/download_beatmaps.py`

## Conventions
- Outputs live under `data/osu2mir/audio`, `data/osu2mir/osz_cache`, and `data/osu2mir/osu2beat2025_metered_beats` (gitignored).
- Use `additional_tools/` scripts for evaluation or metadata extraction as needed.
- Dataset details live in `data/osu2mir/README.md`.

## Do not
- Commit downloaded audio, cached .osz files, or dataset zips.
