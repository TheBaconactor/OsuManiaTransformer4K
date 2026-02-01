# Repo Navigation

This repo is a small monorepo. If you’re lost, start here:
- Project goals + roadmap: `roadmap.md`
- Dataset building: `docs/DATASETS.md`
- Training/export workflow: `docs/TRAINING_SITE.md`

## Directory map

- `modules/beat_this/`
  - Owns: beat tracker model + inference + ONNX export
  - Entrypoint: `modules/beat_this/beat_to_osu.py`
  - Docs: `modules/beat_this/README.md`, `modules/beat_this/docs/beat_to_osu_debug.md`

- `data/osu2mir_audio/`
  - Owns: osu2mir audio downloader + API ID fetch helpers
  - Entrypoints:
    - `data/osu2mir_audio/download_beatmaps.py`
    - `data/osu2mir_audio/fetch_beatmapset_ids.py`

- `data/datasets2/`
  - Owns: osu exports / `.osz` → 4K dataset builder
  - Entrypoints:
    - `data/datasets2/osz_to_dataset.py`
    - `data/datasets2/download_osz_by_ids.py`
    - `data/datasets2/clean_and_classify.py`

- `training/scripts/`
  - Owns: dataset conversion + Ear fine-tune scripts
  - Entrypoints:
    - `training/scripts/build_beat_this_osu_dataset.py`
    - `training/scripts/finetune_beat_this_osu.py`

- `training/wsl/`
  - Owns: WSL2 training runner + ROCm/WSL diagnostics
  - Entrypoints:
    - `training/wsl/run_wsl_ear_finetune.ps1`
    - `training/wsl/diagnose_rocm_wsl.sh`
    - `training/wsl/fix_wsl.ps1`

- `data/osu2mir/`
  - Owns: upstream dataset tooling + tables (reference / research)

## Suggested VS Code setup (optional)

If Explorer is cluttered, add `.vscode/settings.json` with excludes for gitignored datasets and caches.

## Shortcut runner (optional)

If you prefer a single entrypoint, use `tasks.ps1` from repo root:

```powershell
.\tasks.ps1 help
```
