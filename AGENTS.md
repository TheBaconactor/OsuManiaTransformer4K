# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** repo root and below.
- **Key directories:**
  - Modules/beat_this/ (Beat This model, inference, training utilities)
  - Datasets/ (osu2mir audio downloader)
  - Datasets2/ (osu exports -> 4K dataset builder)
  - osu2mir/ (Osu2MIR dataset tools)
  - scripts/ (repo-level dataset + fine-tune scripts)
  - training_site/ (WSL2 training scripts + ROCm helpers)
  - docs/ (training-site notes)

## Modules / subprojects
| Module | Type | Path | What it owns | How to run | Tests | Docs | AGENTS |
|--------|------|------|--------------|------------|-------|------|--------|
| beat_this | python-ml | `Modules/beat_this/` | beat tracker model, inference CLI, Lightning training | `python Modules/beat_this/beat_to_osu.py <audio> --output timing_points.txt` | `python -m pytest Modules/beat_this/tests/test_inference.py` | `Modules/beat_this/README.md` | `Modules/beat_this/AGENTS.md` |
| scripts | python-scripts | `scripts/` | dataset build + osu fine-tune helpers | `python scripts/build_beat_this_osu_dataset.py` | n/a | `docs/TRAINING_SITE.md` | `scripts/AGENTS.md` |
| training_site | wsl-shell | `training_site/` | WSL2/ROCm training runner + diagnostics | `powershell -ExecutionPolicy Bypass -File training_site/run_wsl_ear_finetune.ps1` | n/a | `docs/TRAINING_SITE.md` | `training_site/AGENTS.md` |
| datasets2 | data-prep | `Datasets2/` | osu exports -> dataset (audio + annotations) | `python Datasets2/osz_to_dataset.py` | n/a | n/a | `Datasets2/AGENTS.md` |
| datasets | data-prep | `Datasets/` | osu2mir audio downloader | `python Datasets/download_beatmaps.py --client-id ... --client-secret ...` | n/a | n/a | `Datasets/AGENTS.md` |
| osu2mir | data-prep | `osu2mir/` | Osu2MIR dataset build tools | `python osu2mir/data_conversion.py` | n/a | `osu2mir/README.md` | `osu2mir/AGENTS.md` |

## Cross-domain workflows
- Datasets2 -> Beat This training: `Datasets2/osz_to_dataset.py` writes audio/annotations; `scripts/build_beat_this_osu_dataset.py` converts to `Modules/beat_this/data` (gitignored) for training.
- Fine-tuning: run `scripts/finetune_beat_this_osu.py` (Windows CPU/CUDA) or `training_site/run_ear_finetune.sh` (WSL2/ROCm) to produce checkpoints/ONNX.
- Inference: `Modules/beat_this/beat_to_osu.py` loads `.ckpt` or `beat_this/onnx_models/*.onnx` (DirectML) and writes timing points.
- Osu2MIR flow: `Datasets/download_beatmaps.py` fetches beatmaps; `osu2mir/data_partition.py` + `osu2mir/data_conversion.py` build the dataset.

## Verification (preferred commands)
- Keep runs narrow; heavy training stays in WSL2. Suggested smoke checks:
  - `python -m pytest Modules/beat_this/tests/test_inference.py`
  - `python scripts/build_beat_this_osu_dataset.py --max-items 1 --no-spectrograms` (requires Datasets2 data)

## Docs usage
- Do not open/read `docs/` unless the task requires it or the user asks.

## Global conventions
- Large artifacts are gitignored; do not commit datasets, audio, checkpoints, or ONNX files.
- Prefer running Python scripts from repo root; use absolute paths in scripts only when required.
- For AMD GPU training, use WSL2 workflows in `training_site/` (see `docs/TRAINING_SITE.md`).

## Project anchor: The Four Foundations
- **The Ear (Audio Analysis):** timing, rhythm, phrasing, emphasis, and when not to place notes.
- **The Eye (Review + Taste):** critique/justify style choices; match patterns to mood/intent.
- **The Brain (Contextual Generation):** long-range coherence across sections, motifs, density ramps.
- **The Hands (Playability + Iteration):** ergonomics, readability, difficulty fairness, playtest feedback loop.

## Do not
- Run long training jobs on Windows unless explicitly requested.
- Add generated dataset or model outputs to git.

## Links to module instructions
- `Modules/beat_this/AGENTS.md`
- `scripts/AGENTS.md`
- `training_site/AGENTS.md`
- `Datasets2/AGENTS.md`
- `Datasets/AGENTS.md`
- `osu2mir/AGENTS.md`
