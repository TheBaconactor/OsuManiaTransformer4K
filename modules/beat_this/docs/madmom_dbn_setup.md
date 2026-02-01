# madmom DBN setup (Windows)

Beat This! can optionally run a classic DBN/Viterbi-style beat tracker from `madmom` as a postprocessing step. In this repo, `modules/beat_this/beat_to_osu.py` exposes it via `--postprocess dbn` (or `--elastic-dbn`).

## Why this needs a separate setup

- `madmom` is not readily installable on Python 3.11+ on Windows via `pip` (it builds native extensions).
- On Windows, the most reliable path is often a dedicated Conda env with Python 3.10 and installing `madmom` from Git via `pip` (it builds native extensions).

Note: on some machines, installing from `git+https://github.com/CPJKU/madmom.git` can also work directly in a repo-local `venv` (Python 3.11+), but it may still require build tooling.

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
pip install -r modules/beat_this/requirements.txt
pip install -e modules/beat_this
```

4) Run timing generation with DBN postprocessing:
```bash
python modules/beat_this/beat_to_osu.py data/datasets2/audio/-1_0_terceS_Hard.mp3 ^
  --method legacy --legacy-mode anchor --legacy-anchor-ms 3000 ^
  --postprocess dbn --force-meter 4 --device cpu ^
  -o data/datasets2/timing_points/-1_0_terceS_Hard.legacy_anchor_3s.dbn.timing.txt
```

## Fallback behavior

If you run `--postprocess dbn` without `madmom` installed, `modules/beat_this/beat_to_osu.py` prints a warning and falls back to `--postprocess minimal`.
