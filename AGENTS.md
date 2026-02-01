# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** repo root and below.
- **Key directories:**
  - modules/ (code modules)
  - data/ (dataset tooling + downloaded artifacts)
  - training/ (fine-tune scripts + WSL2/ROCm runner)
  - tools/ (analysis utilities)
  - docs/ (notes + workflows)

## Modules / subprojects
| Module | Type | Path | What it owns | How to run | Tests | Docs | AGENTS |
|--------|------|------|--------------|------------|-------|------|--------|
| beat_this | python-ml | `modules/beat_this/` | beat tracker model, inference CLI, Lightning training | `python modules/beat_this/beat_to_osu.py <audio> --output timing_points.txt` | `python -m pytest modules/beat_this/tests/test_inference.py` | `modules/beat_this/README.md` | `modules/beat_this/AGENTS.md` |
| scripts | python-scripts | `training/scripts/` | dataset build + fine-tune helpers | `python training/scripts/build_beat_this_osu_dataset.py` | n/a | `docs/TRAINING_SITE.md` | `training/scripts/AGENTS.md` |
| wsl | wsl-shell | `training/wsl/` | WSL2/ROCm training runner + diagnostics | `powershell -ExecutionPolicy Bypass -File training/wsl/run_wsl_ear_finetune.ps1` | n/a | `docs/TRAINING_SITE.md` | `training/wsl/AGENTS.md` |
| datasets2 | data-prep | `data/datasets2/` | osu exports -> 4K dataset (audio + annotations) | `python data/datasets2/osz_to_dataset.py` | n/a | `docs/DATASETS.md` | `data/datasets2/AGENTS.md` |
| osu2mir_audio | data-prep | `data/osu2mir_audio/` | osu! API v2 search + downloader | `python data/osu2mir_audio/fetch_beatmapset_ids.py` | n/a | `docs/DATASETS.md` | `data/osu2mir_audio/AGENTS.md` |
| osu2mir | data-prep | `data/osu2mir/` | upstream Osu2MIR dataset tools | `python data/osu2mir/data_conversion.py` | n/a | `data/osu2mir/README.md` | `data/osu2mir/AGENTS.md` |

## Cross-domain workflows
- Datasets2 -> Beat This training: `data/datasets2/osz_to_dataset.py` writes audio/annotations; `training/scripts/build_beat_this_osu_dataset.py` converts to `modules/beat_this/data` (gitignored) for training.
- Fine-tuning: run `training/scripts/finetune_beat_this_osu.py` (Windows CPU/CUDA) or `training/wsl/run_ear_finetune.sh` (WSL2/ROCm) to produce checkpoints/ONNX.
- Inference: `modules/beat_this/beat_to_osu.py` loads `.ckpt` or `beat_this/onnx_models/*.onnx` (DirectML) and writes timing points.
- Osu2MIR flow: `data/osu2mir_audio/fetch_beatmapset_ids.py` builds ID lists; `data/osu2mir_audio/download_beatmaps.py` downloads audio/.osz cache.

## Verification (preferred commands)
- Keep runs narrow; heavy training stays in WSL2. Suggested smoke checks:
  - `python -m pytest modules/beat_this/tests/test_inference.py`
  - `python training/scripts/build_beat_this_osu_dataset.py --max-items 1 --no-spectrograms` (requires `data/datasets2`)

## Docs usage
- Do not open/read `docs/` unless the task requires it or the user asks.

## Global conventions
- Large artifacts are gitignored; do not commit datasets, audio, checkpoints, or ONNX files.
- Prefer running Python scripts from repo root; use absolute paths in scripts only when required.
- For AMD GPU training, use WSL2 workflows in `training/wsl/` (see `docs/TRAINING_SITE.md`).

## Project anchor: The Four Foundations
- **The Ear (Audio Analysis):** timing, rhythm, phrasing, emphasis, and when not to place notes.
- **The Eye (Review + Taste):** critique/justify style choices; match patterns to mood/intent.
- **The Brain (Contextual Generation):** long-range coherence across sections, motifs, density ramps.
- **The Hands (Playability + Iteration):** ergonomics, readability, difficulty fairness, playtest feedback loop.

## Do not
- Run long training jobs on Windows unless explicitly requested.
- Add generated dataset or model outputs to git.

## Links to module instructions
- `modules/beat_this/AGENTS.md`
- `training/scripts/AGENTS.md`
- `training/wsl/AGENTS.md`
- `data/datasets2/AGENTS.md`
- `data/osu2mir_audio/AGENTS.md`
- `data/osu2mir/AGENTS.md`
