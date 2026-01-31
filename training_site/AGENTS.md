# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `training_site/` and below.
- **Key scripts:**
  - run_wsl_ear_finetune.ps1 (Windows entrypoint)
  - run_ear_finetune.sh (WSL2 runner)
  - diagnose_rocm_wsl.sh (ROCm sanity checks)

## Commands
- **Windows entrypoint:** `powershell -ExecutionPolicy Bypass -File training_site/run_wsl_ear_finetune.ps1`
- **WSL2 runner:** `./training_site/run_ear_finetune.sh`
- **ROCm diagnostics (WSL2):** `./training_site/diagnose_rocm_wsl.sh`

## Conventions
- Training should run inside a WSL2 ext4 repo (not `/mnt/c/...`) for performance.
- Outputs ONNX to `Modules/beat_this/beat_this/onnx_models/` (gitignored).
- Keep scripts POSIX-friendly; prefer LF line endings for `.sh` files.

## Do not
- Run long training on Windows unless explicitly requested.
- Commit ONNX exports or training artifacts.
