# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `modules/beat_this/` and below.
- **Key directories:**
  - beat_this/ (package code)
  - launch_scripts/ (training/eval utilities)
  - tests/ (pytest)
  - docs/ (module docs)

## Commands
- **Install (editable):** `pip install -e modules/beat_this`
- **Inference:** `python modules/beat_this/beat_to_osu.py <audio> --output timing_points.txt`
- **Tests:** `python -m pytest modules/beat_this/tests/test_inference.py`

## Conventions
- `beat_this/onnx_models/*.onnx` and `modules/beat_this/data` are generated (gitignored).
- DirectML inference on Windows uses `--device dml` and `--checkpoint <name>` where `<name>` maps to an ONNX file.
- Training workflows live in `training/scripts/` and `training/wsl/` to keep this module focused on inference/model code.

## Common pitfalls
- If you use `--checkpoint <name>`, ensure `beat_this/onnx_models/<name>.onnx` exists.
- DBN postprocessing requires `madmom` (see `docs/madmom_dbn_setup.md`).

## Do not
- Commit model outputs, ONNX files, or generated datasets.
