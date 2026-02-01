from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


COMMON_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)


@dataclass(frozen=True)
class RedLine:
    offset_ms: float
    ms_per_beat: float
    meter: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _load_annotation(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mapper_redlines(ann: dict) -> list[RedLine]:
    tps = ann.get("timing_points", [])
    red = [tp for tp in tps if tp.get("uninherited")]
    red.sort(key=lambda tp: float(tp["offset"]))
    return [
        RedLine(
            offset_ms=float(tp["offset"]),
            ms_per_beat=float(tp["beat_length"]),
            meter=int(tp.get("meter", 4)),
        )
        for tp in red
    ]


def _load_hit_times_ms(ann: dict) -> np.ndarray:
    hos = ann.get("hit_objects", [])
    times = [int(ho["time"]) for ho in hos]
    times.sort()
    return np.asarray(times, dtype=np.float64)


def _active_redline(redlines: list[RedLine], t_ms: float) -> RedLine:
    if not redlines:
        raise ValueError("No redlines")
    lo, hi = 0, len(redlines)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if redlines[mid].offset_ms <= t_ms:
            lo = mid
        else:
            hi = mid
    return redlines[lo]


def _snap_error_ms(
    t_ms: float,
    red: RedLine,
    divisors: tuple[int, ...] = COMMON_DIVISORS,
) -> float:
    dt = t_ms - red.offset_ms
    best_err = float("inf")
    for div in divisors:
        step = red.ms_per_beat / float(div)
        if step <= 0:
            continue
        k = round(dt / step)
        snapped = red.offset_ms + k * step
        err = abs(t_ms - snapped)
        if err < best_err:
            best_err = err
    return best_err


def snapping_metrics(hit_times_ms: np.ndarray, redlines: list[RedLine]) -> dict[str, float]:
    errors = np.empty(hit_times_ms.size, dtype=np.float64)
    for i, t in enumerate(hit_times_ms):
        red = _active_redline(redlines, float(t))
        errors[i] = _snap_error_ms(float(t), red)
    return {
        "median_ms": float(np.median(errors)),
        "p95_ms": float(np.percentile(errors, 95)),
        "mean_ms": float(np.mean(errors)),
        "max_ms": float(np.max(errors)),
        "le2": float(np.mean(errors <= 2.0)),
        "le5": float(np.mean(errors <= 5.0)),
        "le10": float(np.mean(errors <= 10.0)),
    }


def _interp_prob(prob: np.ndarray, t_ms: np.ndarray, fps: float) -> np.ndarray:
    t = np.asarray(t_ms, dtype=np.float64) * (fps / 1000.0)
    i0 = np.floor(t).astype(np.int64)
    frac = t - i0
    i0 = np.clip(i0, 0, len(prob) - 1)
    i1 = np.clip(i0 + 1, 0, len(prob) - 1)
    return (1.0 - frac) * prob[i0] + frac * prob[i1]


def audio_gridfit_score(
    *,
    beat_prob: np.ndarray,
    redlines: list[RedLine],
    start_ms: float,
    end_ms: float,
    fps: int = 50,
) -> float:
    """
    Proxy "audio alignment" score: average beat probability at gridline times.

    Higher is better. Note this uses Beat This activations as the reference signal.
    """
    if not redlines:
        return float("nan")
    beat_prob = np.asarray(beat_prob, dtype=np.float64).reshape(-1)
    total = 0.0
    weight = 0.0
    for i, red in enumerate(redlines):
        seg_start = max(start_ms, red.offset_ms)
        seg_end = end_ms
        if i + 1 < len(redlines):
            seg_end = min(seg_end, redlines[i + 1].offset_ms)
        if seg_end <= seg_start or red.ms_per_beat <= 0:
            continue
        n0 = int(np.floor((seg_start - red.offset_ms) / red.ms_per_beat))
        n1 = int(np.ceil((seg_end - red.offset_ms) / red.ms_per_beat))
        n = np.arange(n0, n1 + 1, dtype=np.int64)
        grid_ms = red.offset_ms + n.astype(np.float64) * red.ms_per_beat
        grid_ms = grid_ms[(grid_ms >= seg_start) & (grid_ms <= seg_end)]
        if grid_ms.size < 8:
            continue
        p = _interp_prob(beat_prob, grid_ms, fps=float(fps))
        seg_score = float(np.mean(p))
        seg_weight = float(seg_end - seg_start)
        total += seg_score * seg_weight
        weight += seg_weight
    return total / weight if weight > 0 else float("nan")


def predict_gridfit_redline(
    *,
    audio_path: Path,
    device: str,
) -> tuple[list[RedLine], np.ndarray, float]:
    """
    Run Beat This framewise inference and infer a single redline with gridfit.
    Returns (redlines, beat_prob, seconds).
    """
    import sys as _sys

    _sys.path.insert(0, str((Path(__file__).resolve().parent / "Modules" / "beat_this")))
    import beat_to_osu as b

    t0 = time.perf_counter()
    beat_times, downbeat_times, beat_logits, _ = b.get_frames_and_beats(
        str(audio_path), device=device, refine=True
    )
    offset_ms, mpb = b._fit_timing_grid_from_logits(
        beat_logits=np.asarray(beat_logits, dtype=np.float64),
        beat_times_s=np.asarray(beat_times, dtype=np.float64),
        fps=50,
        prefer_faster=True,
    )
    meter = b._meter_from_downbeats(
        downbeats_s=np.asarray(downbeat_times, dtype=np.float64),
        offset_ms=float(offset_ms),
        ms_per_beat=float(mpb),
    )
    t1 = time.perf_counter()
    red = [RedLine(offset_ms=float(round(offset_ms)), ms_per_beat=float(mpb), meter=int(meter))]
    beat_prob = _sigmoid(np.asarray(beat_logits, dtype=np.float64))
    return red, beat_prob, (t1 - t0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset2 scorecard: mapper vs predicted timing points."
    )
    parser.add_argument(
        "--device",
        default="dml",
        help="Inference device for prediction (cpu/cuda/dml). Default: %(default)s",
    )
    parser.add_argument(
        "--out",
        default="reports/datasets2_scorecard.csv",
        help="CSV output path (default: %(default)s).",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    ann_dir = repo / "data" / "datasets2" / "annotations"
    audio_dir = repo / "data" / "datasets2" / "audio"
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        ann = _load_annotation(ann_path)
        audio_file = ann.get("audio_file")
        if not audio_file:
            continue
        audio_path = audio_dir / str(audio_file)
        if not audio_path.exists():
            continue

        mapper_red = _load_mapper_redlines(ann)
        hit_times = _load_hit_times_ms(ann)

        mapper_snap = snapping_metrics(hit_times, mapper_red)

        pred_red, beat_prob, seconds = predict_gridfit_redline(
            audio_path=audio_path, device=args.device
        )
        pred_snap = snapping_metrics(hit_times, pred_red)

        start_ms = float(hit_times.min()) if hit_times.size else 0.0
        end_ms = float(hit_times.max()) if hit_times.size else 0.0
        mapper_audio_score = audio_gridfit_score(
            beat_prob=beat_prob,
            redlines=mapper_red,
            start_ms=start_ms,
            end_ms=end_ms,
            fps=50,
        )
        pred_audio_score = audio_gridfit_score(
            beat_prob=beat_prob,
            redlines=pred_red,
            start_ms=start_ms,
            end_ms=end_ms,
            fps=50,
        )

        # Basic BPM/offset deltas for single-redline maps.
        if mapper_red:
            m0 = mapper_red[0]
            p0 = pred_red[0]
            delta_offset = p0.offset_ms - m0.offset_ms
            delta_mpb = p0.ms_per_beat - m0.ms_per_beat
        else:
            delta_offset = float("nan")
            delta_mpb = float("nan")

        rows.append(
            {
                "song_id": ann.get("song_id", ann_path.stem),
                "annotation": ann_path.name,
                "audio": audio_path.name,
                "mapper_red_count": len(mapper_red),
                "pred_red_count": len(pred_red),
                "mapper_median_ms": mapper_snap["median_ms"],
                "mapper_p95_ms": mapper_snap["p95_ms"],
                "pred_median_ms": pred_snap["median_ms"],
                "pred_p95_ms": pred_snap["p95_ms"],
                "pred_le5": pred_snap["le5"],
                "delta_offset_ms": delta_offset,
                "delta_mpb_ms": delta_mpb,
                "mapper_audio_score": mapper_audio_score,
                "pred_audio_score": pred_audio_score,
                "pred_seconds": seconds,
            }
        )

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Print a tiny summary.
    if rows:
        pred_med = np.array([r["pred_median_ms"] for r in rows], dtype=np.float64)
        pred_p95 = np.array([r["pred_p95_ms"] for r in rows], dtype=np.float64)
        print(f"Wrote {report_path} ({len(rows)} rows)")
        print(
            "Pred snapping: median(median_ms)=%.3f  median(p95_ms)=%.3f"
            % (float(np.median(pred_med)), float(np.median(pred_p95)))
        )
    else:
        print("No rows written (missing annotations/audio?)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
