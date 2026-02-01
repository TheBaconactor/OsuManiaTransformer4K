# training_site (legacy compatibility)

This repo’s canonical WSL/ROCm training scripts live in `training/wsl/`.

This folder exists only to keep older commands working (it forwards to `training/wsl/*`):
- `training_site/fix_wsl.ps1` → `training/wsl/fix_wsl.ps1`
- `training_site/run_wsl_ear_finetune.ps1` → `training/wsl/run_wsl_ear_finetune.ps1`
- `training_site/run_ear_finetune.sh` → `training/wsl/run_ear_finetune.sh`
- `training_site/diagnose_rocm_wsl.sh` → `training/wsl/diagnose_rocm_wsl.sh`

