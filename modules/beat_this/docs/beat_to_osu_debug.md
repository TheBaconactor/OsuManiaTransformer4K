# `beat_to_osu.py` debug + “mapper intent” refinement

This repo’s `modules/beat_this/beat_to_osu.py` can optionally dump raw model outputs (for debugging) and refine timing points using note times (to approximate mapper intent).

## Timing profiles (recommended)

To avoid juggling lots of low-level flags, use `--timing-profile`:

- `stable` (default): metronomic / produced music (minimal redlines)
- `live`: live / rubato recordings (adds frequent resync redlines). Mapper heuristic: consider placing redlines roughly every **bar line** (or even every **half-measure**) to keep long recordings in sync without over-jittering.
- `auto`: decide per-audio (gridfit vs resync)

## Debug bundle

Use `--dump-dir <folder>` to write intermediate artifacts:

- `frames.npz`: `beat_logits`, `downbeat_logits`, plus sigmoid’d `beat_prob`, `downbeat_prob` (fps=50)
- `beats_s.npy`, `downbeats_s.npy`: detected beat times in seconds
- `bundle.json`: arguments + timing points + any refinement report
- Optional: `--dump-spectrogram` writes `logmel_f128xT.npy` (float16)

Example:
```bash
python modules/beat_this/beat_to_osu.py data/datasets2/audio/-1_0_terceS_Hard.mp3 ^
  --timing-profile live --device dml ^
  --dump-dir reports/debug/etude
```

## Using a fine-tuned checkpoint

`beat_to_osu.py` supports `--checkpoint` / `--beat-this-checkpoint`:

```bash
python modules/beat_this/beat_to_osu.py data/datasets2/audio/-1_0_terceS_Hard.mp3 ^
  --checkpoint modules/beat_this/checkpoints/osu_finetune/last.ckpt ^
  --timing-profile live -o out.timing.txt
```

Notes:
- DirectML/ONNX inference (`--device dml`) only works with exported ONNX models; custom `.ckpt` will fall back to CPU unless you export ONNX.
- Training/export workflow (WSL2 training → ONNX → Windows inference): see `docs/TRAINING_SITE.md`.

## Note-aware timing refinement (“mapper intent”)

Humans place redlines to make **hitobjects quantize cleanly**, not to match every audible onset. To mimic this, pass a JSON containing `hit_objects[].time` (ms):

```bash
python modules/beat_this/beat_to_osu.py data/datasets2/audio/-1_0_terceS_Hard.mp3 ^
  --timing-profile live --device dml ^
  --hitobjects-json data/datasets2/annotations/-1_0_terceS_Hard.json ^
  --intent-offset-step-ms 1 --intent-mpb-search-pct 0.01 ^
  -o data/datasets2/timing_points/-1_0_terceS_Hard.auto.intent.timing.txt
```

Notes:
- This is an **upper bound** when you use human hitobjects. For real inference, pass your **predicted** note times instead.
- By default it only adjusts section phase (offset). You can also search tempo with `--intent-mpb-search-pct`.
