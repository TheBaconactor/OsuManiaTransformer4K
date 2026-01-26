# madmom DBN setup (Windows)

Beat This! can optionally run a classic DBN/Viterbi-style beat tracker from `madmom` as a postprocessing step. In this repo, `Modules/beat_this/beat_to_osu.py` exposes it via `--postprocess dbn` (or `--elastic-dbn`).

## Why this needs a separate setup

- `madmom` is not readily installable on Python 3.11+ on Windows via `pip` (it builds native extensions).
- On Windows, the most reliable path is a dedicated Conda env with Python 3.10 and installing `madmom` from Git via `pip` (it builds a native wheel).

## Recommended: Conda env (Python 3.10)

From the repo root:

1) Create and activate an environment:
```bash
conda create -n beat-this-dbn python=3.10 -y
conda activate beat-this-dbn
```
If `conda` is not on your `PATH`, run it via its full path (e.g. `C:\\Users\\<you>\\miniconda3\\Scripts\\conda.exe`).

2) Install `madmom` (builds native extensions):
```bash
pip install "numpy<2" cython
pip install "madmom @ git+https://github.com/CPJKU/madmom.git"
```

3) Install Beat This! inference deps:
```bash
pip install -r Modules/beat_this/requirements.txt
pip install -e Modules/beat_this
```

4) Run timing generation with DBN postprocessing:
```bash
python Modules/beat_this/beat_to_osu.py Datasets2/audio/-1_0_terceS_Hard.mp3 ^
  --method legacy --legacy-mode anchor --legacy-anchor-ms 3000 ^
  --postprocess dbn --force-meter 4 --device cpu ^
  -o Datasets2/timing_points/-1_0_terceS_Hard.legacy_anchor_3s.dbn.timing.txt
```

## Fallback behavior

If you run `--postprocess dbn` without `madmom` installed, `Modules/beat_this/beat_to_osu.py` prints a warning and falls back to `--postprocess viterbi`.
