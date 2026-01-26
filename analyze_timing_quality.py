from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


COMMON_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)


@dataclass(frozen=True)
class RedLine:
    offset_ms: float
    ms_per_beat: float
    meter: int


def _load_annotation_redlines(annotation_path: Path) -> list[RedLine]:
    ann = json.loads(annotation_path.read_text(encoding="utf-8"))
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


def _load_annotation_hit_objects(annotation_path: Path) -> np.ndarray:
    ann = json.loads(annotation_path.read_text(encoding="utf-8"))
    hos = ann.get("hit_objects", [])
    times = [int(ho["time"]) for ho in hos]
    times.sort()
    return np.asarray(times, dtype=np.float64)


def _load_timing_file_redlines(timing_path: Path) -> list[RedLine]:
    redlines: list[RedLine] = []
    for line in timing_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        redlines.append(
            RedLine(
                offset_ms=float(parts[0]),
                ms_per_beat=float(parts[1]),
                meter=int(parts[2]),
            )
        )
    redlines.sort(key=lambda x: x.offset_ms)
    return redlines


def _active_redline(redlines: list[RedLine], t_ms: float) -> RedLine:
    # last redline whose offset <= t_ms
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
) -> tuple[float, int]:
    dt = t_ms - red.offset_ms
    best_err = float("inf")
    best_div = divisors[0]
    for div in divisors:
        step = red.ms_per_beat / float(div)
        if step <= 0:
            continue
        k = round(dt / step)
        snapped = red.offset_ms + k * step
        err = abs(t_ms - snapped)
        if err < best_err:
            best_err = err
            best_div = div
    return best_err, best_div


def evaluate_snapping(
    hit_times_ms: np.ndarray,
    redlines: list[RedLine],
) -> dict[str, object]:
    if len(redlines) == 0:
        raise ValueError("No red timing points provided")
    if hit_times_ms.size == 0:
        raise ValueError("No hit objects found")

    errors = np.empty(hit_times_ms.size, dtype=np.float64)
    best_divs = np.empty(hit_times_ms.size, dtype=np.int64)
    for i, t in enumerate(hit_times_ms):
        red = _active_redline(redlines, float(t))
        err, div = _snap_error_ms(float(t), red)
        errors[i] = err
        best_divs[i] = div

    out: dict[str, object] = {
        "n": int(errors.size),
        "mean_ms": float(errors.mean()),
        "median_ms": float(np.median(errors)),
        "p95_ms": float(np.percentile(errors, 95)),
        "max_ms": float(errors.max()),
        "le_1ms": int((errors <= 1.0).sum()),
        "le_2ms": int((errors <= 2.0).sum()),
        "le_5ms": int((errors <= 5.0).sum()),
        "le_10ms": int((errors <= 10.0).sum()),
        "le_20ms": int((errors <= 20.0).sum()),
    }

    # Divisor usage histogram (useful to see if we're forcing weird snapping)
    uniq, cnt = np.unique(best_divs, return_counts=True)
    out["divisor_hist"] = {int(k): int(v) for k, v in zip(uniq, cnt)}
    return out


def _print_report(label: str, stats: dict[str, object]) -> None:
    n = stats["n"]
    print(
        f"{label}: n={n} median={stats['median_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms "
        f"mean={stats['mean_ms']:.3f}ms max={stats['max_ms']:.3f}ms "
        f"<=2ms={stats['le_2ms']}/{n} <=5ms={stats['le_5ms']}/{n} <=10ms={stats['le_10ms']}/{n}"
    )
    print(f"  divisor_hist={stats['divisor_hist']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate timing points by how well they snap the map's hit objects."
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to Datasets2 annotation JSON (contains timing_points + hit_objects).",
    )
    parser.add_argument(
        "--timing",
        action="append",
        default=[],
        help="Path to generated timing points file (osu [TimingPoints] text). Can be passed multiple times.",
    )
    args = parser.parse_args()

    ann_path = Path(args.annotation)
    hit_times = _load_annotation_hit_objects(ann_path)
    mapper_red = _load_annotation_redlines(ann_path)

    mapper_stats = evaluate_snapping(hit_times, mapper_red)
    _print_report("Mapper", mapper_stats)

    for timing in args.timing:
        timing_path = Path(timing)
        red = _load_timing_file_redlines(timing_path)
        stats = evaluate_snapping(hit_times, red)
        _print_report(timing_path.name, stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
