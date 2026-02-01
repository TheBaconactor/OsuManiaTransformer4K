from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MatchStats:
    n_a: int
    n_b: int
    matched: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    within_2ms: int
    within_5ms: int
    within_10ms: int
    within_20ms: int


def _nearest_deltas_ms(a_s: np.ndarray, b_s: np.ndarray) -> np.ndarray:
    a_s = np.asarray(a_s, dtype=np.float64)
    b_s = np.asarray(b_s, dtype=np.float64)
    if a_s.size == 0 or b_s.size == 0:
        return np.array([], dtype=np.float64)

    # Both sequences are sorted times; do a two-pointer nearest-neighbor match
    # from A to B (not one-to-one, just nearest in B for each in A).
    deltas = np.empty(a_s.size, dtype=np.float64)
    j = 0
    for i, t in enumerate(a_s):
        while j + 1 < b_s.size and abs(b_s[j + 1] - t) <= abs(b_s[j] - t):
            j += 1
        deltas[i] = abs(b_s[j] - t) * 1000.0
    return deltas


def _summarize(a_s: np.ndarray, b_s: np.ndarray) -> MatchStats:
    deltas = _nearest_deltas_ms(a_s, b_s)
    if deltas.size == 0:
        return MatchStats(
            n_a=int(np.asarray(a_s).size),
            n_b=int(np.asarray(b_s).size),
            matched=0,
            mean_ms=float("nan"),
            median_ms=float("nan"),
            p95_ms=float("nan"),
            max_ms=float("nan"),
            within_2ms=0,
            within_5ms=0,
            within_10ms=0,
            within_20ms=0,
        )

    return MatchStats(
        n_a=int(np.asarray(a_s).size),
        n_b=int(np.asarray(b_s).size),
        matched=int(deltas.size),
        mean_ms=float(np.mean(deltas)),
        median_ms=float(np.median(deltas)),
        p95_ms=float(np.percentile(deltas, 95)),
        max_ms=float(np.max(deltas)),
        within_2ms=int(np.sum(deltas <= 2.0)),
        within_5ms=int(np.sum(deltas <= 5.0)),
        within_10ms=int(np.sum(deltas <= 10.0)),
        within_20ms=int(np.sum(deltas <= 20.0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Beat This! outputs: PyTorch CPU vs ONNX Runtime DirectML."
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file (mp3/ogg/wav).",
    )
    parser.add_argument(
        "--checkpoint",
        default="final0",
        help="Beat This checkpoint name/path/url (default: %(default)s).",
    )
    parser.add_argument(
        "--provider",
        default="DmlExecutionProvider",
        help="ONNX Runtime provider (default: %(default)s).",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    # Import from repo copy.
    import sys

    sys.path.insert(0, str((Path(__file__).resolve().parent / "Modules" / "beat_this")))

    from beat_this.inference import File2Beats
    from beat_this.onnx_inference import OnnxFile2Beats

    cpu = File2Beats(checkpoint_path=args.checkpoint, device="cpu", dbn=False)
    onnx = OnnxFile2Beats(
        checkpoint_path=args.checkpoint,
        provider=args.provider,
        dbn=False,
    )

    cpu_beats, cpu_downbeats = cpu(audio_path)
    onnx_beats, onnx_downbeats = onnx(audio_path)

    cpu_beats = np.asarray(cpu_beats, dtype=np.float64)
    cpu_downbeats = np.asarray(cpu_downbeats, dtype=np.float64)
    onnx_beats = np.asarray(onnx_beats, dtype=np.float64)
    onnx_downbeats = np.asarray(onnx_downbeats, dtype=np.float64)

    beat_stats = _summarize(cpu_beats, onnx_beats)
    downbeat_stats = _summarize(cpu_downbeats, onnx_downbeats)

    def fmt(s: MatchStats) -> str:
        if s.matched == 0:
            return f"n_cpu={s.n_a} n_onnx={s.n_b} (no matches)"
        return (
            f"n_cpu={s.n_a} n_onnx={s.n_b} "
            f"mean={s.mean_ms:.3f}ms median={s.median_ms:.3f}ms p95={s.p95_ms:.3f}ms max={s.max_ms:.3f}ms "
            f"<=2ms={s.within_2ms}/{s.matched} <=5ms={s.within_5ms}/{s.matched} "
            f"<=10ms={s.within_10ms}/{s.matched} <=20ms={s.within_20ms}/{s.matched}"
        )

    print(f"Audio: {audio_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX provider: {args.provider}")
    print("")
    print("Beats   :", fmt(beat_stats))
    print("Downbeats:", fmt(downbeat_stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
