# Repo Navigation

This repo is a small monorepo. If you’re lost, start here:
- Project goals + roadmap: `roadmap.md`
- Dataset building: `docs/DATASETS.md`
- Training/export workflow: `docs/TRAINING_SITE.md`

## Directory map

- `Modules/beat_this/`
  - Owns: beat tracker model + inference + ONNX export
  - Entrypoint: `Modules/beat_this/beat_to_osu.py`
  - Docs: `Modules/beat_this/README.md`, `Modules/beat_this/docs/beat_to_osu_debug.md`

- `Datasets/`
  - Owns: osu2mir audio downloader + API ID fetch helpers
  - Entrypoints:
    - `Datasets/download_beatmaps.py`
    - `Datasets/fetch_beatmapset_ids.py`

- `Datasets2/`
  - Owns: osu exports / `.osz` → 4K dataset builder
  - Entrypoints:
    - `Datasets2/osz_to_dataset.py`
    - `Datasets2/download_osz_by_ids.py`
    - `Datasets2/clean_and_classify.py`

- `scripts/`
  - Owns: dataset conversion + Ear fine-tune scripts
  - Entrypoints:
    - `scripts/build_beat_this_osu_dataset.py`
    - `scripts/finetune_beat_this_osu.py`

- `training_site/`
  - Owns: WSL2 training runner + ROCm/WSL diagnostics
  - Entrypoints:
    - `training_site/run_wsl_ear_finetune.ps1`
    - `training_site/diagnose_rocm_wsl.sh`
    - `training_site/fix_wsl.ps1`

- `osu2mir/`
  - Owns: upstream dataset tooling + tables (reference / research)

## Suggested VS Code setup (optional)

If Explorer is cluttered, add `.vscode/settings.json` with excludes for gitignored datasets and caches.

## Shortcut runner (optional)

If you prefer a single entrypoint, use `tasks.ps1` from repo root:

```powershell
.\tasks.ps1 help
```
