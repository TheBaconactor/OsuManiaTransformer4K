"""
Beat This! to Osu! Timing Point Converter

This script uses the SOTA 'Beat This!' model to detect beats in an audio file
and converts them into osu! .osu format timing points.

Usage:
    python beat_to_osu.py path/to/audio.mp3
    python beat_to_osu.py path/to/audio.mp3 --output timing_points.txt
    python beat_to_osu.py path/to/audio.mp3 --timing-profile live --output timing_points.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TIMING_PROFILES: dict[str, dict[str, object]] = {
    # Metronomic / produced music: prefer a single BPM + offset.
    "stable": {
        "method": "gridfit",
        "postprocess": "minimal",
        "legacy_mode": "change",
        "legacy_anchor_ms": 4000.0,
        "legacy_anchor_alpha": 0.2,
        "refine": False,
        "logit_refine": False,
        "onset_refine": False,
    },
    # Rubato / live recordings: prevent long-term drift by re-anchoring regularly.
    "live": {
        "method": "legacy",
        "postprocess": "dbn",
        "legacy_mode": "anchor",
        "legacy_anchor_ms": 3000.0,
        "legacy_anchor_alpha": 0.25,
        "refine": True,
        "logit_refine": False,
        "onset_refine": False,
    },
    # Try to decide if gridfit is sufficient; otherwise resync.
    "auto": {
        "method": "auto",
        "postprocess": "minimal",
        "legacy_mode": "change",
        "legacy_anchor_ms": 4000.0,
        "legacy_anchor_alpha": 0.2,
        "refine": False,
        "logit_refine": False,
        "onset_refine": False,
    },
}


def _timing_profile_from_argv(argv: list[str]) -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--timing-profile",
        "--profile",
        dest="timing_profile",
        choices=sorted(TIMING_PROFILES.keys()),
        default="stable",
    )
    args, _ = pre.parse_known_args(argv)
    return str(args.timing_profile)


def _defaults_for_timing_profile(profile: str) -> dict[str, object]:
    if profile not in TIMING_PROFILES:
        raise ValueError(
            f"Unknown timing profile {profile!r}; expected one of {sorted(TIMING_PROFILES.keys())}"
        )
    return dict(TIMING_PROFILES[profile])


def _gridfit_score_segment(
    beat_prob: np.ndarray,
    *,
    offset_ms: float,
    ms_per_beat: float,
    seg_start_ms: float,
    seg_end_ms: float,
    fps: float,
    min_gridlines: int = 8,
) -> float:
    """
    Score a timing grid by how well its beat gridlines land on high beat probability.
    Higher is better.
    """
    mpb = float(ms_per_beat)
    if not np.isfinite(mpb) or mpb <= 0:
        return -float("inf")
    seg_start_ms = float(seg_start_ms)
    seg_end_ms = float(seg_end_ms)
    if not (np.isfinite(seg_start_ms) and np.isfinite(seg_end_ms)) or seg_end_ms <= seg_start_ms:
        return -float("inf")

    # Enumerate beat gridlines in this segment.
    n0 = int(np.floor((seg_start_ms - float(offset_ms)) / mpb))
    n1 = int(np.ceil((seg_end_ms - float(offset_ms)) / mpb))
    n = np.arange(n0, n1 + 1, dtype=np.int64)
    grid_ms = float(offset_ms) + n.astype(np.float64) * mpb
    grid_ms = grid_ms[(grid_ms >= seg_start_ms) & (grid_ms < seg_end_ms)]
    if grid_ms.size < int(min_gridlines):
        return -float("inf")

    p = _interp_prob(beat_prob, grid_ms, fps=float(fps))
    return float(np.mean(p))


def refine_timing_points_with_logits(
    timing_points: list[dict],
    beat_logits: np.ndarray,
    *,
    fps: float = 50.0,
    offset_step_ms: float = 0.5,
    offset_window_ms: float = 25.0,
    mpb_search_pct: float = 0.01,
    mpb_search_steps: int = 7,
    min_gridlines: int = 8,
) -> tuple[list[dict], list[dict]]:
    """
    Audio-aware refinement: adjust each redline's phase (and slightly its tempo) to better
    align beat gridlines to Beat This activations.

    This is "unsupervised" (no hitobjects) and is mainly meant to reduce phase bias when
    timing points are produced from noisy beat picks.
    """
    beat_logits = np.asarray(beat_logits, dtype=np.float64).reshape(-1)
    beat_prob = _sigmoid(beat_logits)
    if beat_prob.size < 8:
        return timing_points, []

    tps = [dict(tp) for tp in timing_points]
    tps.sort(key=lambda tp: float(tp.get("offset_ms", 0.0)))

    audio_end_ms = (float(beat_prob.size - 1) / float(fps)) * 1000.0
    report: list[dict] = []

    step = float(offset_step_ms)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("offset_step_ms must be > 0")

    pct = float(max(0.0, mpb_search_pct))
    n_steps = int(max(1, mpb_search_steps))
    if n_steps == 1:
        pct = 0.0

    for i, tp in enumerate(tps):
        base_offset = float(tp["offset_ms"])
        base_mpb = float(tp["ms_per_beat"])
        if not np.isfinite(base_offset) or not np.isfinite(base_mpb) or base_mpb <= 0:
            report.append({"i": int(i), "refined": False, "reason": "invalid_base"})
            continue

        seg_start = max(0.0, base_offset)
        seg_end = (
            float(tps[i + 1]["offset_ms"]) if (i + 1) < len(tps) else float(audio_end_ms)
        )
        seg_end = min(float(audio_end_ms), float(seg_end))
        if seg_end <= seg_start:
            report.append({"i": int(i), "refined": False, "reason": "empty_segment"})
            continue

        # Keep ordering stable.
        min_offset = float(tps[i - 1]["offset_ms"]) + 1e-3 if i > 0 else -float("inf")
        max_offset = float(seg_end) - 1e-3

        win = float(offset_window_ms)
        if not np.isfinite(win) or win <= 0:
            win = 0.0
        win = min(win, 0.49 * float(base_mpb))
        offsets = np.arange(base_offset - win, base_offset + win + step, step, dtype=np.float64)
        offsets = np.clip(offsets, min_offset, max_offset)
        offsets = np.unique(offsets)

        if pct > 0.0:
            factors = np.linspace(1.0 - pct, 1.0 + pct, n_steps, dtype=np.float64)
            mpb_candidates = base_mpb * factors
        else:
            mpb_candidates = np.asarray([base_mpb], dtype=np.float64)

        best_score = -float("inf")
        best_offset = base_offset
        best_mpb = base_mpb

        base_score = _gridfit_score_segment(
            beat_prob,
            offset_ms=base_offset,
            ms_per_beat=base_mpb,
            seg_start_ms=seg_start,
            seg_end_ms=seg_end,
            fps=float(fps),
            min_gridlines=int(min_gridlines),
        )

        for mpb in mpb_candidates:
            mpb = float(mpb)
            if not np.isfinite(mpb) or mpb <= 0:
                continue
            for off in offsets:
                score = _gridfit_score_segment(
                    beat_prob,
                    offset_ms=float(off),
                    ms_per_beat=float(mpb),
                    seg_start_ms=seg_start,
                    seg_end_ms=seg_end,
                    fps=float(fps),
                    min_gridlines=int(min_gridlines),
                )
                if score > best_score:
                    best_score = float(score)
                    best_offset = float(off)
                    best_mpb = float(mpb)

        # If we couldn't score this segment (too short), keep as-is.
        if not np.isfinite(best_score):
            report.append({"i": int(i), "refined": False, "reason": "no_score"})
            continue

        tp["offset_ms"] = float(best_offset)
        tp["ms_per_beat"] = round(float(best_mpb), 6)

        report.append(
            {
                "i": int(i),
                "refined": True,
                "offset_ms_before": float(base_offset),
                "offset_ms_after": float(best_offset),
                "mpb_before": float(base_mpb),
                "mpb_after": float(best_mpb),
                "score_before": float(base_score),
                "score_after": float(best_score),
            }
        )

    # Ensure monotonic offsets after float updates; if collisions happen, keep first.
    tps.sort(key=lambda tp: float(tp.get("offset_ms", 0.0)))
    dedup: list[dict] = []
    seen = set()
    for tp in tps:
        off_ms = float(tp["offset_ms"])
        key = int(round(off_ms * 1000.0))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(tp)

    return dedup, report


def detect_audio_onsets_ms(
    audio_path: str,
    *,
    n_fft: int = 512,
    hop_length: int = 128,
    peak_width_frames: int = 3,
    peak_percentile: float = 0.90,
    min_separation_ms: float = 50.0,
    max_onsets: int | None = 6000,
) -> np.ndarray:
    """
    Lightweight, dependency-free onset detector (spectral flux peak picking).

    Returns onset timestamps in milliseconds (float64), sorted ascending.
    """
    from beat_this.preprocessing import load_audio

    signal, sr = load_audio(str(audio_path))
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.size < int(n_fft):
        return np.asarray([], dtype=np.float64)

    n_fft = int(n_fft)
    hop_length = int(hop_length)
    if n_fft <= 0 or hop_length <= 0:
        return np.asarray([], dtype=np.float64)

    n_frames = 1 + (x.size - n_fft) // hop_length
    if n_frames <= 1:
        return np.asarray([], dtype=np.float64)

    # Frame as a strided view to avoid copies.
    stride = x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(hop_length * stride, stride),
        writeable=False,
    )
    window = np.hanning(n_fft).astype(np.float32)
    frames_w = frames * window[None, :]

    spec = np.fft.rfft(frames_w, axis=1)
    mag = np.abs(spec).astype(np.float32)

    # Spectral flux (positive differences).
    diff = mag[1:] - mag[:-1]
    diff = np.maximum(diff, 0.0)
    flux = diff.sum(axis=1).astype(np.float32)
    flux = np.concatenate([np.asarray([0.0], dtype=np.float32), flux])

    # Normalize to [0, 1] (avoid all-zero).
    flux = flux - float(np.min(flux))
    mx = float(np.max(flux))
    if mx > 0:
        flux = flux / mx

    width = max(1, int(peak_width_frames))
    mask = _local_maxima_mask(flux, width=width)
    if not np.any(mask):
        return np.asarray([], dtype=np.float64)

    q = float(np.clip(float(peak_percentile), 0.0, 1.0))
    thresh = float(np.quantile(flux, q))
    peaks = np.where(mask & (flux >= thresh))[0]
    if peaks.size == 0:
        return np.asarray([], dtype=np.float64)

    # Enforce a minimum time separation between peaks.
    min_sep_frames = int(round((float(min_separation_ms) / 1000.0) * float(sr) / float(hop_length)))
    min_sep_frames = max(1, min_sep_frames)
    kept: list[int] = []
    last = -10**12
    for idx in peaks.tolist():
        if idx - last >= min_sep_frames:
            kept.append(int(idx))
            last = int(idx)

    # Optional cap: keep strongest peaks.
    if max_onsets is not None and len(kept) > int(max_onsets):
        order = sorted(kept, key=lambda i: float(flux[i]), reverse=True)[: int(max_onsets)]
        kept = sorted(order)

    times_ms = (np.asarray(kept, dtype=np.float64) * float(hop_length) / float(sr)) * 1000.0
    return times_ms


def get_device(requested: str) -> str:
    """
    Resolve the requested device to an actual PyTorch device string.
    
    Supports: cpu, cuda, dml (DirectML for AMD GPUs)
    """
    import torch
    
    if requested == "dml":
        try:
            import torch_directml
            # Return as string for compatibility with libraries that expect strings/ints
            return str(torch_directml.device())
        except ImportError:
            print("[WARN] torch-directml not installed. Falling back to CPU.", file=sys.stderr)
            return "cpu"
    elif requested.startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            return requested
        else:
            # Try DirectML as fallback for AMD GPUs
            try:
                import torch_directml
                print("[INFO] CUDA not available. Using DirectML (AMD GPU).", file=sys.stderr)
                return str(torch_directml.device())
            except ImportError:
                print("[WARN] Neither CUDA nor DirectML available. Using CPU.", file=sys.stderr)
                return "cpu"
    else:
        return requested


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _interp_prob(prob: np.ndarray, t_ms: np.ndarray, fps: float) -> np.ndarray:
    """
    Sample a framewise probability curve at arbitrary timestamps (ms) via linear interpolation.
    """
    t = np.asarray(t_ms, dtype=np.float64) * (fps / 1000.0)
    i0 = np.floor(t).astype(np.int64)
    frac = t - i0
    i0 = np.clip(i0, 0, len(prob) - 1)
    i1 = np.clip(i0 + 1, 0, len(prob) - 1)
    return (1.0 - frac) * prob[i0] + frac * prob[i1]


def _quadratic_refine_frames(prob: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """
    Sub-frame quadratic peak interpolation around *given* frame indices.

    This is a safe refinement for beat trackers that output frame indices (e.g. our viterbi),
    because it can't jump to a different peak—only adds a small delta in [-0.5, 0.5] frames.
    """
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    f0 = np.asarray(frames, dtype=np.int64).reshape(-1)
    if p.size < 3 or f0.size == 0:
        return f0.astype(np.float64)

    f = np.clip(f0, 1, int(p.size - 2))
    y0 = p[f - 1]
    y1 = p[f]
    y2 = p[f + 1]
    denom = (y0 - 2.0 * y1 + y2)
    delta = np.zeros_like(y1, dtype=np.float64)
    m = denom != 0.0
    delta[m] = 0.5 * (y0[m] - y2[m]) / denom[m]
    delta = np.clip(delta, -0.5, 0.5)
    return f.astype(np.float64) + delta


def _load_hit_objects_ms(annotation_json_path: str) -> np.ndarray:
    ann = json.loads(Path(annotation_json_path).read_text(encoding="utf-8"))
    hit_objects = ann.get("hit_objects", [])
    times = []
    for ho in hit_objects:
        try:
            times.append(float(ho["time"]))
        except Exception:
            continue
    times.sort()
    return np.asarray(times, dtype=np.float64)


def _snapping_error_for_offsets(
    hit_times_ms: np.ndarray,
    offsets_ms: np.ndarray,
    *,
    ms_per_beat: float,
    divisors: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
) -> np.ndarray:
    """
    Compute mean snapping error for each candidate offset (ms).

    For each hit object, we allow snapping to multiple common divisors and take the minimum error.
    """
    hit_times_ms = np.asarray(hit_times_ms, dtype=np.float64).reshape(1, -1)
    offsets_ms = np.asarray(offsets_ms, dtype=np.float64).reshape(-1, 1)
    mpb = float(ms_per_beat)
    if hit_times_ms.size == 0 or offsets_ms.size == 0 or not np.isfinite(mpb) or mpb <= 0:
        return np.asarray([], dtype=np.float64)

    best = np.full((offsets_ms.shape[0], hit_times_ms.shape[1]), np.inf, dtype=np.float64)
    for div in divisors:
        step = mpb / float(div)
        if not np.isfinite(step) or step <= 0:
            continue
        dt = hit_times_ms - offsets_ms
        k = np.round(dt / step)
        snapped = offsets_ms + k * step
        err = np.abs(hit_times_ms - snapped)
        best = np.minimum(best, err)

    return np.mean(best, axis=1)


def refine_timing_points_with_hitobjects(
    timing_points: list[dict],
    hit_times_ms: np.ndarray,
    *,
    offset_step_ms: float = 1.0,
    max_offset_shift_ms: float | None = None,
    mpb_search_pct: float = 0.0,
    mpb_search_steps: int = 5,
    objective: str = "mean",
    min_hits_per_section: int = 8,
    divisors: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
) -> tuple[list[dict], list[dict]]:
    """
    Note-aware refinement: adjust each redline's phase (and optionally tempo) to better quantize note times.

    This mimics what human mappers actually optimize: clean snapping of hit objects to simple subdivisions.

    Returns (refined_timing_points, per_section_report).
    """
    hit_times_ms = np.asarray(hit_times_ms, dtype=np.float64).reshape(-1)
    hit_times_ms = hit_times_ms[np.isfinite(hit_times_ms)]
    hit_times_ms.sort()

    if hit_times_ms.size == 0 or len(timing_points) == 0:
        return timing_points, []

    tps = [dict(tp) for tp in timing_points]
    tps.sort(key=lambda tp: float(tp.get("offset_ms", 0.0)))

    if objective not in {"mean"}:
        raise ValueError(f"objective must be 'mean' for now, got {objective!r}")

    step = float(offset_step_ms)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("offset_step_ms must be > 0")

    if int(mpb_search_steps) < 1:
        raise ValueError("mpb_search_steps must be >= 1")
    if int(mpb_search_steps) == 1:
        mpb_search_pct = 0.0

    report: list[dict] = []
    last_hit = float(hit_times_ms[-1])

    for i, tp in enumerate(tps):
        base_offset = float(tp["offset_ms"])
        base_mpb = float(tp["ms_per_beat"])
        seg_start = base_offset
        seg_end = float(tps[i + 1]["offset_ms"]) if (i + 1) < len(tps) else (last_hit + 2000.0)

        # Keep ordering stable.
        min_offset = float(tps[i - 1]["offset_ms"]) + 1.0 if i > 0 else -float("inf")
        max_offset = seg_end - 1.0

        hits = hit_times_ms[(hit_times_ms >= seg_start) & (hit_times_ms < seg_end)]
        if hits.size < int(min_hits_per_section) or not np.isfinite(base_mpb) or base_mpb <= 0:
            report.append(
                {
                    "i": int(i),
                    "refined": False,
                    "hits": int(hits.size),
                    "offset_ms_before": float(tp["offset_ms"]),
                    "mpb_before": float(tp["ms_per_beat"]),
                }
            )
            continue

        # Build MPB candidates around the current estimate.
        pct = float(max(0.0, mpb_search_pct))
        if pct > 0:
            factors = np.linspace(1.0 - pct, 1.0 + pct, int(mpb_search_steps), dtype=np.float64)
            mpb_candidates = base_mpb * factors
        else:
            mpb_candidates = np.asarray([base_mpb], dtype=np.float64)

        best_cost = float("inf")
        best_offset = base_offset
        best_mpb = base_mpb

        for mpb in mpb_candidates:
            if not np.isfinite(mpb) or mpb <= 0:
                continue
            half = float(mpb) * 0.5
            if max_offset_shift_ms is not None:
                m = float(max_offset_shift_ms)
                if np.isfinite(m) and m > 0:
                    half = min(half, m)
            deltas = np.arange(-half, half + step, step, dtype=np.float64)
            offsets = base_offset + deltas
            offsets = np.clip(offsets, min_offset, max_offset)
            # De-duplicate after clipping.
            offsets = np.unique(offsets)
            costs = _snapping_error_for_offsets(hits, offsets, ms_per_beat=float(mpb), divisors=divisors)
            if costs.size == 0:
                continue
            j = int(np.argmin(costs))
            cost = float(costs[j])
            if cost < best_cost:
                best_cost = cost
                best_offset = float(offsets[j])
                best_mpb = float(mpb)

        tps[i]["offset_ms"] = round(best_offset)
        tps[i]["ms_per_beat"] = round(best_mpb, 6)

        report.append(
            {
                "i": int(i),
                "refined": True,
                "hits": int(hits.size),
                "cost_mean_ms": float(best_cost),
                "offset_ms_before": float(base_offset),
                "offset_ms_after": float(tps[i]["offset_ms"]),
                "mpb_before": float(base_mpb),
                "mpb_after": float(tps[i]["ms_per_beat"]),
            }
        )

    # Ensure monotonic offsets after rounding; if collisions happen, keep first.
    tps.sort(key=lambda tp: float(tp.get("offset_ms", 0.0)))
    dedup: list[dict] = []
    seen = set()
    for tp in tps:
        off = int(tp["offset_ms"])
        if off in seen:
            continue
        seen.add(off)
        dedup.append(tp)

    return dedup, report


def dump_debug_bundle(
    dump_dir: Path,
    *,
    audio_path: str,
    fps: int,
    beat_times_s: np.ndarray,
    downbeat_times_s: np.ndarray,
    beat_logits: np.ndarray,
    downbeat_logits: np.ndarray,
    timing_points: list[dict],
    meta: dict,
    dump_spectrogram: bool = False,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)

    beat_logits = np.asarray(beat_logits, dtype=np.float32).reshape(-1)
    downbeat_logits = np.asarray(downbeat_logits, dtype=np.float32).reshape(-1)
    beat_prob = _sigmoid(beat_logits.astype(np.float64)).astype(np.float32)
    downbeat_prob = _sigmoid(downbeat_logits.astype(np.float64)).astype(np.float32)

    np.savez_compressed(
        dump_dir / "frames.npz",
        fps=np.asarray([int(fps)], dtype=np.int32),
        beat_logits=beat_logits,
        downbeat_logits=downbeat_logits,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
    )

    np.save(dump_dir / "beats_s.npy", np.asarray(beat_times_s, dtype=np.float64))
    np.save(dump_dir / "downbeats_s.npy", np.asarray(downbeat_times_s, dtype=np.float64))

    bundle = {
        "audio_path": str(audio_path),
        "fps": int(fps),
        "meta": meta,
        "timing_points": timing_points,
    }
    (dump_dir / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    if dump_spectrogram:
        try:
            import torch
            import soxr
            from beat_this.preprocessing import LogMelSpect, load_audio

            signal, sr = load_audio(audio_path, dtype="float64")
            if sr != 22050:
                signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
                sr = 22050
            x = torch.tensor(signal, dtype=torch.float32, device="cpu")
            mel = LogMelSpect(device="cpu")(x).detach().cpu().numpy().astype(np.float16)
            np.save(dump_dir / "logmel_f128xT.npy", mel)
        except Exception as e:
            (dump_dir / "spectrogram_error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")

def _local_maxima_mask(x: np.ndarray, width: int) -> np.ndarray:
    """
    Boolean mask of local maxima within +/- width samples (inclusive).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.asarray([], dtype=bool)
    m = x.copy()
    for k in range(1, int(width) + 1):
        m = np.maximum(m, np.roll(x, k))
        m = np.maximum(m, np.roll(x, -k))
    mask = x >= m
    # Invalidate wrap-around artifacts introduced by roll.
    if width > 0 and x.size > width:
        mask[:width] = False
        mask[-width:] = False
    return mask


def _viterbi_beats_from_prob(
    *,
    beat_prob: np.ndarray,
    fps: int,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    transition_lambda: float = 100.0,
    tempo_prior_lambda: float = 30.0,
    start_window_s: float = 8.0,
    end_window_s: float = 8.0,
    peak_width_frames: int = 3,
    peak_threshold: float = 0.2,
) -> np.ndarray:
    """
    Lightweight DBN-like beat tracking via Viterbi on (time, interval) states.

    Returns beat frame indices (int64), strictly increasing.
    """
    p = np.asarray(beat_prob, dtype=np.float64).reshape(-1)
    if p.size < 4:
        return np.asarray([], dtype=np.int64)

    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    t = int(p.size)

    d_min = int(np.floor((60.0 * float(fps)) / float(max_bpm)))
    d_max = int(np.ceil((60.0 * float(fps)) / float(min_bpm)))
    d_min = max(1, d_min)
    d_max = max(d_min + 1, d_max)

    # Tempo prior center from short-lag autocorrelation of beat activations.
    # This reduces half/double-time ambiguities by preferring the strongest periodicity.
    corr_scores = []
    for d in range(d_min, d_max + 1):
        corr_scores.append(float(np.sum(p[d:] * p[:-d])))
    corr_scores = np.asarray(corr_scores, dtype=np.float64)
    best = float(np.max(corr_scores))
    # Prefer faster tempo among near-ties.
    tie = np.where(corr_scores >= best * 0.99)[0]
    d0 = int((d_min + int(tie.min())) if tie.size else (d_min + int(np.argmax(corr_scores))))

    # Only consider transitions that land on reasonably confident local maxima.
    confident_peak_mask = _local_maxima_mask(p, width=int(peak_width_frames)) & (
        p >= float(peak_threshold)
    )
    peak_mask = confident_peak_mask.copy()

    # If peaks are too sparse, fall back to considering all frames.
    if int(np.sum(peak_mask)) < 16:
        peak_mask = np.ones(t, dtype=bool)

    # Start/end anchors (avoid "single beat" solutions).
    sw = int(max(1, round(float(start_window_s) * float(fps))))
    ew = int(max(1, round(float(end_window_s) * float(fps))))
    start_hi = min(t, sw)
    end_lo = max(0, t - ew)

    def _pick_anchor_frame(*, lo: int, hi: int, take_last: bool) -> int:
        lo = int(max(0, lo))
        hi = int(min(t, hi))
        if hi <= lo:
            return int(np.clip(lo, 0, t - 1))

        peaks = np.flatnonzero(confident_peak_mask[lo:hi]) + lo
        if peaks.size == 0:
            return int(lo + int(np.argmax(p[lo:hi])))
        return int(peaks[-1] if take_last else peaks[0])

    start_frame = _pick_anchor_frame(lo=0, hi=start_hi, take_last=False)
    end_frame = _pick_anchor_frame(lo=end_lo, hi=t, take_last=True)
    if end_frame <= start_frame:
        start_frame = _pick_anchor_frame(lo=0, hi=min(t, max(4, start_hi * 2)), take_last=False)
        end_frame = _pick_anchor_frame(lo=max(0, t - max(4, ew * 2)), hi=t, take_last=True)
        if end_frame <= start_frame:
            start_frame = 0
            end_frame = int(t - 1)

    # Viterbi DP: dp[frame, d] where d in [d_min, d_max].
    d_vals = np.arange(d_min, d_max + 1, dtype=np.int64)
    d_count = int(d_vals.size)
    dp = np.full((t, d_count), np.inf, dtype=np.float64)
    back_d = np.full((t, d_count), -1, dtype=np.int16)

    log_d = np.log(d_vals.astype(np.float64))
    log_d0 = float(np.log(float(d0)))

    # Initialize at start_frame.
    start_cost = -np.log(p[start_frame])
    dp[start_frame, :] = start_cost

    for frame in range(start_frame + 1, t):
        if not bool(peak_mask[frame]):
            continue
        emit = -np.log(p[frame])
        for di, d in enumerate(d_vals):
            prev_frame = frame - int(d)
            if prev_frame < start_frame:
                continue
            prev_costs = dp[prev_frame, :]
            if not np.isfinite(prev_costs).any():
                continue
            # Penalize log-tempo changes.
            trans = float(transition_lambda) * (log_d[di] - log_d) ** 2
            prior = float(tempo_prior_lambda) * (log_d[di] - log_d0) ** 2
            total = prev_costs + trans + prior
            best_prev = int(np.argmin(total))
            best_cost = float(total[best_prev])
            dp[frame, di] = emit + best_cost
            back_d[frame, di] = np.int16(best_prev)

    # Pick best end state near end_frame (allow small wiggle).
    end_search = np.arange(max(start_frame + 1, end_frame - d_max), min(t, end_frame + d_max + 1))
    best_end = None
    best_cost = float("inf")
    for frame in end_search:
        if not bool(peak_mask[frame]):
            continue
        row = dp[frame, :]
        if not np.isfinite(row).any():
            continue
        di = int(np.argmin(row))
        cost = float(row[di])
        # Prefer solutions that end closer to end_frame if costs are similar.
        cost += 1e-6 * abs(frame - end_frame)
        if cost < best_cost:
            best_cost = cost
            best_end = (frame, di)

    if best_end is None:
        return np.asarray([], dtype=np.int64)

    # Backtrack.
    beats = []
    frame, di = best_end
    while frame >= start_frame and di >= 0:
        beats.append(int(frame))
        d = int(d_vals[di])
        prev_frame = frame - d
        prev_di = int(back_d[frame, di])
        if prev_frame < start_frame or prev_di < 0:
            break
        frame, di = prev_frame, prev_di

    beats.append(int(start_frame))
    beats = np.unique(np.asarray(beats, dtype=np.int64))
    beats.sort()
    return beats


def _downbeats_from_beats_and_logits(
    *,
    beat_frames: np.ndarray,
    downbeat_prob: np.ndarray,
    meter: int = 4,
) -> np.ndarray:
    beat_frames = np.asarray(beat_frames, dtype=np.int64)
    if beat_frames.size == 0:
        return np.asarray([], dtype=np.int64)
    p = np.asarray(downbeat_prob, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return np.asarray([], dtype=np.int64)
    m = int(meter) if int(meter) > 0 else 4
    m = max(1, min(m, 12))

    idx = beat_frames[(beat_frames >= 0) & (beat_frames < p.size)]
    if idx.size == 0:
        return np.asarray([], dtype=np.int64)

    # Choose phase that best aligns downbeat activations.
    phase_scores = []
    for ph in range(m):
        phase_scores.append(float(np.sum(p[idx[ph::m]])))
    best_phase = int(np.argmax(phase_scores))
    return idx[best_phase::m]


def _fit_timing_grid_from_logits(
    *,
    beat_logits: np.ndarray,
    beat_times_s: np.ndarray,
    fps: int = 50,
    prefer_faster: bool = True,
    offset_search_step_ms: float = 1.0,
) -> tuple[float, float]:
    """
    Infer (offset_ms, ms_per_beat) by fitting a periodic comb to the beat probability curve.

    Returns:
        (offset_ms, ms_per_beat)
    """
    beat_times_s = np.asarray(beat_times_s, dtype=np.float64)
    if beat_times_s.size < 2:
        raise ValueError("Need at least 2 beats to fit timing grid")

    beat_logits = np.asarray(beat_logits, dtype=np.float64).reshape(-1)
    prob = _sigmoid(beat_logits)

    beats_ms = beat_times_s * 1000.0
    ibis = np.diff(beats_ms)
    base_mpb = float(np.median(ibis))
    if not np.isfinite(base_mpb) or base_mpb <= 0:
        raise ValueError("Invalid base beat period from beats")

    # Candidate beat lengths: base, double-time, half-time.
    candidates = [base_mpb, base_mpb / 2.0, base_mpb * 2.0]
    # Filter to a reasonable osu-relevant BPM band but keep wide enough.
    candidates = [mpb for mpb in candidates if 180.0 <= mpb <= 1500.0]
    if not candidates:
        candidates = [base_mpb]

    start_ms = max(0.0, float(beats_ms[0] - 2.0 * base_mpb))
    end_ms = float(beats_ms[-1] + 2.0 * base_mpb)

    def score(mpb: float, effective_mpb: float, phase_ms: float) -> float:
        # Anchor the absolute offset close to the first detected beat.
        k = np.floor((beats_ms[0] - phase_ms) / effective_mpb)
        offset_ms = phase_ms + k * effective_mpb
        n0 = int(np.floor((start_ms - offset_ms) / effective_mpb))
        n1 = int(np.ceil((end_ms - offset_ms) / effective_mpb))
        n = np.arange(n0, n1 + 1, dtype=np.int64)
        grid_ms = offset_ms + n.astype(np.float64) * effective_mpb
        grid_ms = grid_ms[(grid_ms >= start_ms) & (grid_ms <= end_ms)]
        if grid_ms.size < 8:
            return -1.0

        # Sample probabilities exactly at gridline times.
        p = _interp_prob(prob, grid_ms, fps=float(fps))
        return float(np.mean(p))

    best: tuple[float, float, int, float] | None = None  # (score, mpb, step_mult, phase_ms)
    for mpb in candidates:
        # Only allow skipping every other beat for mpb candidates that are not
        # slower than the base estimate; otherwise the scorer would prefer very
        # sparse grids (e.g. 4x slower) just because peaks are stronger.
        step_mults = range(1, 3) if mpb <= base_mpb * 1.05 else range(1, 2)
        for step_mult in step_mults:
            effective_mpb = mpb * float(step_mult)
            if effective_mpb < 180.0 or effective_mpb > 1500.0:
                continue

            step = offset_search_step_ms
            # search phase in [0, effective_mpb)
            phases = np.arange(0.0, effective_mpb, step, dtype=np.float64)
            scores = np.array(
                [score(mpb, effective_mpb, ph) for ph in phases], dtype=np.float64
            )
            idx = int(np.argmax(scores))
            best_phase = float(phases[idx])
            best_score = float(scores[idx])

            # Quadratic interpolation in phase space for sub-ms refinement.
            if 1 <= idx < len(phases) - 1:
                s0, s1, s2 = (
                    float(scores[idx - 1]),
                    float(scores[idx]),
                    float(scores[idx + 1]),
                )
                denom = (s0 - 2.0 * s1 + s2)
                if denom != 0.0:
                    delta = 0.5 * (s0 - s2) / denom
                    if -1.0 <= delta <= 1.0:
                        best_phase = best_phase + delta * step

            cand = (best_score, mpb, step_mult, best_phase)
            if best is None:
                best = cand
                continue

            score_eps = 1e-4
            if cand[0] > best[0] + score_eps:
                best = cand
            elif abs(cand[0] - best[0]) <= score_eps and prefer_faster:
                if cand[1] < best[1]:
                    best = cand

    assert best is not None
    _, mpb, step_mult, phase_ms = best

    # Convert phase to an absolute offset close to first beat (osu timing point offset).
    effective_mpb = mpb * float(step_mult)
    k = np.floor((beats_ms[0] - phase_ms) / effective_mpb)
    offset_ms = phase_ms + k * effective_mpb
    return float(offset_ms), float(mpb)


def _meter_from_downbeats(
    *,
    downbeats_s: np.ndarray,
    offset_ms: float,
    ms_per_beat: float,
) -> int:
    downbeats_s = np.asarray(downbeats_s, dtype=np.float64)
    if downbeats_s.size < 2 or ms_per_beat <= 0:
        return 4
    downbeats_ms = downbeats_s * 1000.0
    idx = np.rint((downbeats_ms - offset_ms) / ms_per_beat).astype(np.int64)
    idx = np.sort(idx)
    gaps = np.diff(idx)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return 4
    from collections import Counter

    meter = Counter(map(int, gaps)).most_common(1)[0][0]
    # osu maps overwhelmingly use 3/4 or 4/4; if the model suggests 6 or 8,
    # prefer the simpler equivalent (6->3, 8->4) since many mappers double BPM
    # without doubling the time signature.
    while meter > 4 and meter % 2 == 0:
        meter //= 2
    return max(1, min(int(meter), 8))


def _gridfit_residuals_ms(
    beat_times_s: np.ndarray,
    ms_per_beat: float,
    phase_search_step_ms: float = 1.0,
) -> np.ndarray:
    beat_times_ms = np.asarray(beat_times_s, dtype=np.float64) * 1000.0
    if beat_times_ms.size == 0 or ms_per_beat <= 0:
        return np.asarray([], dtype=np.float64)

    mpb = float(ms_per_beat)
    # Find the best phase in [0, mpb) that minimizes mean distance to the nearest gridline.
    # This makes the "electronic probe" robust to an arbitrary initial offset choice.
    phases = np.arange(0.0, mpb, float(phase_search_step_ms), dtype=np.float64)
    if phases.size == 0:
        phases = np.array([0.0], dtype=np.float64)

    def residuals_for_phase(phase: float) -> np.ndarray:
        # Distance to nearest multiple of mpb with phase shift.
        # Equivalent to wrapping to [-mpb/2, mpb/2].
        r = np.remainder(beat_times_ms - phase + 0.5 * mpb, mpb) - 0.5 * mpb
        return np.abs(r)

    best_mean = float("inf")
    best_residuals = None
    for ph in phases:
        r = residuals_for_phase(float(ph))
        m = float(np.mean(r)) if r.size else float("inf")
        if m < best_mean:
            best_mean = m
            best_residuals = r

    return (
        best_residuals
        if best_residuals is not None
        else np.asarray([], dtype=np.float64)
    )


def beats_to_timing_points_elastic(
    beats: list[float],
    downbeats: list[float],
    *,
    min_section_beats: int = 16,
    downbeat_match_tolerance_ms: float = 80.0,
    segment_penalty_ms: float = 250.0,
    tempo_change_penalty: float = 20.0,
    max_sections: int = 128,
    meter: int = 4,
) -> list[dict]:
    """
    Elastic (DP/Viterbi-like) piecewise-constant tempo solver.

    Fits segments of constant ms_per_beat to Beat This beat times while penalizing
    excessive segmentation and large tempo changes.

    This is intended for "live" / rubato audio where a single redline (gridfit)
    accumulates phase drift, but naive per-IBI segmentation (legacy) creates
    spammy, jittery timing points.
    """
    if len(beats) < 2:
        return []

    beats_ms = np.asarray(beats, dtype=np.float64) * 1000.0
    if not np.all(np.isfinite(beats_ms)):
        beats_ms = beats_ms[np.isfinite(beats_ms)]
    beats_ms.sort()
    if beats_ms.size < 2:
        return []

    downbeats_ms = np.asarray(downbeats, dtype=np.float64) * 1000.0
    downbeats_ms = downbeats_ms[np.isfinite(downbeats_ms)]
    downbeats_ms.sort()
    downbeats_set = set(int(round(x)) for x in downbeats_ms)

    # Candidate breakpoints: prefer starting new sections at downbeats to keep
    # timing points mapper-friendly.
    break_beat_indices: set[int] = {0, int(beats_ms.size - 1)}
    if downbeats_ms.size:
        for db in downbeats_ms:
            j = int(np.argmin(np.abs(beats_ms - db)))
            if abs(float(beats_ms[j] - db)) <= float(downbeat_match_tolerance_ms):
                break_beat_indices.add(j)

    breakpoints = sorted(break_beat_indices)
    if breakpoints[0] != 0:
        breakpoints.insert(0, 0)
    if breakpoints[-1] != int(beats_ms.size - 1):
        breakpoints.append(int(beats_ms.size - 1))

    # DP over breakpoint list indices.
    m = len(breakpoints)
    dp = np.full(m, np.inf, dtype=np.float64)
    prev = np.full(m, -1, dtype=np.int64)
    last_mpb = np.full(m, np.nan, dtype=np.float64)
    dp[0] = 0.0

    # Cache segment fits (start_bp_idx, end_bp_idx) -> (cost, mpb)
    seg_cost = [[None] * m for _ in range(m)]
    seg_mpb = [[None] * m for _ in range(m)]

    def fit_segment(start_beat_idx: int, end_beat_idx: int) -> tuple[float, float]:
        if end_beat_idx <= start_beat_idx:
            return float("inf"), float("nan")
        y = beats_ms[start_beat_idx : end_beat_idx + 1] - beats_ms[start_beat_idx]
        k = np.arange(y.size, dtype=np.float64)
        denom = float(np.dot(k, k))
        if denom <= 0.0:
            return float("inf"), float("nan")
        mpb = float(np.dot(k, y) / denom)
        if not np.isfinite(mpb) or mpb <= 0:
            return float("inf"), float("nan")
        resid = y - k * mpb
        abs_resid = np.abs(resid)
        # Robust, beat-weighted fit cost (ms).
        fit_cost = float(np.mean(abs_resid) * abs_resid.size)
        return fit_cost, mpb

    for j in range(1, m):
        end_idx = breakpoints[j]
        best_cost = float("inf")
        best_i = -1
        best_mpb = float("nan")
        for i in range(0, j):
            start_idx = breakpoints[i]
            n_beats = end_idx - start_idx
            if n_beats < int(min_section_beats):
                continue

            c = seg_cost[i][j]
            mpb = seg_mpb[i][j]
            if c is None or mpb is None:
                c_fit, mpb_fit = fit_segment(start_idx, end_idx)
                seg_cost[i][j] = c_fit
                seg_mpb[i][j] = mpb_fit
                c = c_fit
                mpb = mpb_fit

            if not np.isfinite(c) or not np.isfinite(mpb):
                continue

            total = float(dp[i] + c + float(segment_penalty_ms))
            if i != 0 and np.isfinite(last_mpb[i]):
                total += float(tempo_change_penalty) * abs(float(mpb) - float(last_mpb[i]))

            if total < best_cost:
                best_cost = total
                best_i = i
                best_mpb = float(mpb)

        # If we cannot satisfy min_section_beats using downbeat breakpoints,
        # fall back to allowing a segment from 0.
        if best_i == -1:
            i = 0
            c_fit, mpb_fit = fit_segment(breakpoints[i], end_idx)
            best_cost = float(dp[i] + c_fit + float(segment_penalty_ms))
            best_i = i
            best_mpb = float(mpb_fit)

        dp[j] = best_cost
        prev[j] = best_i
        last_mpb[j] = best_mpb

    # Reconstruct path.
    path = []
    cur = m - 1
    while cur > 0:
        i = int(prev[cur])
        if i < 0:
            break
        path.append((i, cur))
        cur = i
    path.reverse()

    # Convert to timing points at segment starts.
    timing_points: list[dict] = []
    meter_out = int(meter) if int(meter) > 0 else 4
    for (i, j) in path:
        start_idx = breakpoints[i]
        end_idx = breakpoints[j]
        c_fit, mpb = fit_segment(start_idx, end_idx)
        if not np.isfinite(mpb) or mpb <= 0:
            continue
        timing_points.append(
            {
                "offset_ms": round(float(beats_ms[start_idx])),
                "ms_per_beat": round(float(mpb), 6),
                "meter": meter_out,
            }
        )

    # Hard cap: if we still have too many, merge nearest-tempo neighbors.
    if len(timing_points) > int(max_sections) and len(timing_points) >= 2:
        while len(timing_points) > int(max_sections):
            best_k = None
            best_d = float("inf")
            for k in range(len(timing_points) - 1):
                d = abs(float(timing_points[k + 1]["ms_per_beat"]) - float(timing_points[k]["ms_per_beat"]))
                if d < best_d:
                    best_d = d
                    best_k = k
            if best_k is None:
                break
            # Merge k and k+1 by keeping the earlier offset and averaging mpb.
            a = timing_points[best_k]
            b = timing_points[best_k + 1]
            a_mpb = float(a["ms_per_beat"])
            b_mpb = float(b["ms_per_beat"])
            a["ms_per_beat"] = round(float(0.5 * (a_mpb + b_mpb)), 6)
            timing_points.pop(best_k + 1)

    return timing_points


def get_frames_and_beats(
    audio_path: str,
    device: str = "cpu",
    refine: bool = False,
    postprocess: str = "minimal",
    checkpoint_path: str = "final0",
    viterbi_min_bpm: float = 55.0,
    viterbi_max_bpm: float = 215.0,
    viterbi_transition_lambda: float = 100.0,
    viterbi_tempo_prior_lambda: float = 30.0,
    viterbi_peak_threshold: float = 0.2,
    meter_for_downbeats: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (beat_times_s, downbeat_times_s, beat_logits, downbeat_logits).
    """
    from beat_this.inference import _refine_peak_times
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import load_audio

    resolved_device = get_device(device)
    print(f"[INFO] Using device: {resolved_device}", file=sys.stderr)

    signal, sr = load_audio(audio_path)

    if str(resolved_device).startswith("privateuseone"):
        from beat_this.onnx_inference import OnnxAudio2Frames

        try:
            a2f = OnnxAudio2Frames(
                checkpoint_path=str(checkpoint_path),
                provider="DmlExecutionProvider",
                float16=False,
            )
            beat_logits_t, downbeat_logits_t = a2f(signal, sr)
        except Exception as e:
            print(
                f"[WARN] ONNX DirectML failed ({type(e).__name__}: {e}). Falling back to CPU.",
                file=sys.stderr,
            )
            resolved_device = "cpu"
            from beat_this.inference import Audio2Frames

            a2f = Audio2Frames(
                checkpoint_path=str(checkpoint_path),
                device=resolved_device,
                float16=False,
            )
            beat_logits_t, downbeat_logits_t = a2f(signal, sr)
    else:
        from beat_this.inference import Audio2Frames

        a2f = Audio2Frames(
            checkpoint_path=str(checkpoint_path),
            device=resolved_device,
            float16=False,
        )
        beat_logits_t, downbeat_logits_t = a2f(signal, sr)

    if postprocess not in {"minimal", "dbn", "viterbi"}:
        raise ValueError(f"Unknown postprocess={postprocess!r}")

    did_refine = False

    if postprocess == "viterbi":
        beat_prob = _sigmoid(beat_logits_t.detach().cpu().numpy().reshape(-1))
        downbeat_prob = _sigmoid(downbeat_logits_t.detach().cpu().numpy().reshape(-1))
        beat_frames = _viterbi_beats_from_prob(
            beat_prob=beat_prob,
            fps=50,
            min_bpm=float(viterbi_min_bpm),
            max_bpm=float(viterbi_max_bpm),
            transition_lambda=float(viterbi_transition_lambda),
            tempo_prior_lambda=float(viterbi_tempo_prior_lambda),
            peak_threshold=float(viterbi_peak_threshold),
        )
        downbeat_frames = _downbeats_from_beats_and_logits(
            beat_frames=beat_frames,
            downbeat_prob=downbeat_prob,
            meter=int(meter_for_downbeats),
        )
        beat_frames_f = beat_frames.astype(np.float64)
        if refine:
            beat_frames_f = _quadratic_refine_frames(beat_prob, beat_frames)
            did_refine = True
        beat_time = beat_frames_f / 50.0
        downbeat_time = downbeat_frames.astype(np.float64) / 50.0
    else:
        try:
            post = Postprocessor(type=postprocess, fps=50)
        except ModuleNotFoundError as e:
            if postprocess == "dbn" and e.name == "madmom":
                print(
                    "[WARN] postprocess=dbn requested but 'madmom' is not installed; falling back to viterbi.",
                    file=sys.stderr,
                )
                beat_prob = _sigmoid(beat_logits_t.detach().cpu().numpy().reshape(-1))
                downbeat_prob = _sigmoid(downbeat_logits_t.detach().cpu().numpy().reshape(-1))
                beat_frames = _viterbi_beats_from_prob(
                    beat_prob=beat_prob,
                    fps=50,
                    min_bpm=float(viterbi_min_bpm),
                    max_bpm=float(viterbi_max_bpm),
                    transition_lambda=float(viterbi_transition_lambda),
                    tempo_prior_lambda=float(viterbi_tempo_prior_lambda),
                    peak_threshold=float(viterbi_peak_threshold),
                )
                downbeat_frames = _downbeats_from_beats_and_logits(
                    beat_frames=beat_frames,
                    downbeat_prob=downbeat_prob,
                    meter=int(meter_for_downbeats),
                )
                beat_frames_f = beat_frames.astype(np.float64)
                if refine:
                    beat_frames_f = _quadratic_refine_frames(beat_prob, beat_frames)
                    did_refine = True
                beat_time = beat_frames_f / 50.0
                downbeat_time = downbeat_frames.astype(np.float64) / 50.0
            else:
                raise
        else:
            beat_time, downbeat_time = post(beat_logits_t, downbeat_logits_t)

    if refine:
        # Sub-frame refinement via quadratic interpolation on the activation curve.
        if not did_refine:
            beat_time = _refine_peak_times(beat_logits_t, beat_time, fps=50)
        if len(downbeat_time) > 0 and len(beat_time) > 0:
            beat_time_np = np.asarray(beat_time, dtype=np.float64)
            downbeat_time_np = np.asarray(downbeat_time, dtype=np.float64)
            down_ref = np.empty_like(downbeat_time_np)
            for i, dt in enumerate(downbeat_time_np):
                j = int(np.argmin(np.abs(beat_time_np - dt)))
                down_ref[i] = beat_time_np[j]
            downbeat_time = np.unique(down_ref)

    beat_logits = beat_logits_t.detach().cpu().numpy().reshape(-1)
    downbeat_logits = downbeat_logits_t.detach().cpu().numpy().reshape(-1)
    return (
        np.asarray(beat_time, dtype=np.float64),
        np.asarray(downbeat_time, dtype=np.float64),
        beat_logits,
        downbeat_logits,
    )


def get_beats(
    audio_path: str,
    device: str = "cpu",
    refine: bool = False,
    dbn: bool = False,
    checkpoint_path: str = "final0",
) -> tuple[list[float], list[float]]:
    """
    Run Beat This! inference on an audio file.
    
    Returns:
        Tuple of (beats, downbeats) as lists of timestamps in seconds.
    """
    from beat_this.inference import File2Beats
    from beat_this.onnx_inference import OnnxFile2Beats
    
    resolved_device = get_device(device)
    print(f"[INFO] Using device: {resolved_device}", file=sys.stderr)

    # Prefer ONNX Runtime + DirectML on Windows AMD GPUs.
    if str(resolved_device).startswith("privateuseone"):
        try:
            file2beats = OnnxFile2Beats(
                checkpoint_path=str(checkpoint_path),
                provider="DmlExecutionProvider",
                dbn=bool(dbn),
                refine=refine,
            )
            beats, downbeats = file2beats(audio_path)
            return beats.tolist(), downbeats.tolist()
        except Exception as e:
            print(
                f"[WARN] ONNX DirectML failed ({type(e).__name__}: {e}). Falling back to CPU.",
                file=sys.stderr,
            )

    # Default: PyTorch inference (CPU / CUDA).
    try:
        file2beats = File2Beats(
            checkpoint_path=str(checkpoint_path),
            device=resolved_device,
            dbn=bool(dbn),
            refine=refine,
        )
    except ModuleNotFoundError as e:
        if bool(dbn) and e.name == "madmom":
            print(
                "[WARN] DBN requested but 'madmom' is not installed; falling back to dbn=False.",
                file=sys.stderr,
            )
            file2beats = File2Beats(
                checkpoint_path=str(checkpoint_path),
                device=resolved_device,
                dbn=False,
                refine=refine,
            )
        else:
            raise
    beats, downbeats = file2beats(audio_path)
    return beats.tolist(), downbeats.tolist()


def beats_to_timing_points(
    beats: list[float],
    downbeats: list[float],
    tolerance_ms: float = 2.0,
    min_section_beats: int = 4,
    max_section_beats: int | None = None,
    max_section_ms: float | None = None,
    force_meter: int | None = None,
) -> list[dict]:
    """
    Convert beat timestamps into osu! timing points.
    
    This algorithm:
    1. Calculates inter-beat intervals (IBIs).
    2. Detects where the BPM changes significantly.
    3. Creates a new timing point at each BPM change.
    
    Args:
        beats: List of beat timestamps in seconds.
        downbeats: List of downbeat timestamps in seconds.
        tolerance_ms: Maximum variation in ms to consider beats as "same BPM".
        min_section_beats: Minimum beats before allowing a new timing point.
    
    Returns:
        List of timing point dicts with 'offset_ms', 'ms_per_beat', 'meter'.
    """
    if len(beats) < 2:
        return []
    
    beats_ms = [b * 1000.0 for b in beats]
    downbeats_set = set(round(d * 1000.0) for d in downbeats)
    
    # Calculate inter-beat intervals
    ibis = [beats_ms[i+1] - beats_ms[i] for i in range(len(beats_ms) - 1)]
    
    timing_points = []
    section_start_idx = 0
    section_ibis = []
    
    for i, ibi in enumerate(ibis):
        if not section_ibis:
            section_ibis.append(ibi)
            continue
        
        current_avg = np.mean(section_ibis)
        section_len = len(section_ibis)  # number of IBIs in current section
        section_duration_ms = beats_ms[i] - beats_ms[section_start_idx]

        should_resync = False
        if max_section_beats is not None and int(max_section_beats) > 0:
            if section_len >= int(max_section_beats):
                should_resync = True
        if max_section_ms is not None and float(max_section_ms) > 0:
            if float(section_duration_ms) >= float(max_section_ms):
                should_resync = True
        
        # Check if this beat interval deviates too much from the current section
        if (
            (abs(ibi - current_avg) > tolerance_ms or should_resync)
            and section_len >= int(min_section_beats)
        ):
            # Create a timing point for the previous section
            avg_ms_per_beat = np.mean(section_ibis)
            offset_ms = beats_ms[section_start_idx]
            
            # Detect meter (time signature) using downbeats
            if force_meter is not None:
                meter = int(force_meter)
            else:
                meter = detect_meter(beats_ms[section_start_idx:i+1], downbeats_set)
            
            timing_points.append({
                'offset_ms': round(offset_ms),
                'ms_per_beat': round(avg_ms_per_beat, 6),
                'meter': meter
            })
            
            # Start new section
            # Split at the current beat (i), so the new section's first interval
            # (ibi) starts at beats_ms[i] and ends at beats_ms[i+1].
            section_start_idx = i
            section_ibis = [ibi]
        else:
            section_ibis.append(ibi)
    
    # Final section
    if section_ibis:
        avg_ms_per_beat = np.mean(section_ibis)
        offset_ms = beats_ms[section_start_idx]
        if force_meter is not None:
            meter = int(force_meter)
        else:
            meter = detect_meter(beats_ms[section_start_idx:], downbeats_set)
        
        timing_points.append({
            'offset_ms': round(offset_ms),
            'ms_per_beat': round(avg_ms_per_beat, 6),
            'meter': meter
        })
    
    return timing_points


def beats_to_timing_points_anchor_resync(
    beats: list[float],
    downbeats: list[float],
    *,
    anchor_ms: float = 4000.0,
    local_window_ibis: int = 16,
    local_alpha: float = 0.2,
    clamp_ratio: float = 0.15,
    force_meter: int | None = None,
) -> list[dict]:
    """
    "Aggressive resync" timing: place frequent redlines as phase anchors.

    The idea is to avoid per-beat BPM jitter (which looks wildly out-of-sync),
    but still prevent long-term phase drift by re-anchoring the grid every few seconds.
    """
    if len(beats) < 2:
        return []

    beats_ms = np.asarray(beats, dtype=np.float64) * 1000.0
    beats_ms = beats_ms[np.isfinite(beats_ms)]
    beats_ms.sort()
    if beats_ms.size < 2:
        return []

    # Robust global mpb estimate.
    ibis = np.diff(beats_ms)
    mpb0 = float(np.median(ibis))
    if not np.isfinite(mpb0) or mpb0 <= 0:
        return []
    keep = (ibis >= mpb0 * 0.5) & (ibis <= mpb0 * 1.5)
    if np.any(keep):
        mpb0 = float(np.median(ibis[keep]))

    downbeats_set = set(round(d * 1000.0) for d in downbeats)

    if force_meter is not None:
        meter = int(force_meter)
    else:
        meter = 4

    timing_points: list[dict] = []
    current_mpb = float(mpb0)
    start = 0
    timing_points.append(
        {"offset_ms": round(float(beats_ms[start])), "ms_per_beat": round(current_mpb, 6), "meter": meter}
    )

    if anchor_ms <= 0:
        return timing_points

    while start + 1 < beats_ms.size:
        target_t = float(beats_ms[start] + float(anchor_ms))
        nxt = int(np.searchsorted(beats_ms, target_t, side="left"))
        if nxt <= start:
            nxt = start + 1
        if nxt >= beats_ms.size:
            break

        # Optional local tempo adaptation (robust + clamped).
        win_lo = max(start, nxt - int(local_window_ibis))
        win_ibis = np.diff(beats_ms[win_lo : nxt + 1])
        win_ibis = win_ibis[np.isfinite(win_ibis)]
        if win_ibis.size:
            local_mpb = float(np.median(win_ibis))
            if np.isfinite(local_mpb) and local_mpb > 0:
                proposed = (1.0 - float(local_alpha)) * current_mpb + float(local_alpha) * local_mpb
                lo = current_mpb * (1.0 - float(clamp_ratio))
                hi = current_mpb * (1.0 + float(clamp_ratio))
                current_mpb = float(np.clip(proposed, lo, hi))

        timing_points.append(
            {
                "offset_ms": round(float(beats_ms[nxt])),
                "ms_per_beat": round(float(current_mpb), 6),
                "meter": meter,
            }
        )
        start = nxt

    return timing_points


def detect_meter(beat_timestamps_ms: list[float], downbeats_set: set[int]) -> int:
    """
    Detect the time signature meter (beats per measure) from downbeat positions.
    
    Returns:
        The detected meter (e.g., 4 for 4/4 time).
    """
    if len(beat_timestamps_ms) < 2:
        return 4  # Default to 4/4
    
    # Find beats that coincide with downbeats
    downbeat_indices = []
    for i, bt in enumerate(beat_timestamps_ms):
        # Check if this beat is near a downbeat (within 50ms tolerance)
        bt_rounded = round(bt)
        for db in downbeats_set:
            if abs(bt_rounded - db) < 50:
                downbeat_indices.append(i)
                break
    
    if len(downbeat_indices) < 2:
        return 4  # Default
    
    # Calculate beats between consecutive downbeats
    gaps = [downbeat_indices[i+1] - downbeat_indices[i] for i in range(len(downbeat_indices) - 1)]
    
    if not gaps:
        return 4
    
    # Most common gap is the meter
    from collections import Counter
    meter = Counter(gaps).most_common(1)[0][0]
    
    return max(1, min(meter, 8))  # Clamp between 1 and 8


def format_osu_timing_points(timing_points: list[dict]) -> str:
    """
    Format timing points into osu! .osu file format.
    
    Format: Offset,MillisecondsPerBeat,Meter,SampleSet,SampleIndex,Volume,Uninherited,Effects
    """
    lines = ["[TimingPoints]"]
    
    for tp in timing_points:
        # Uninherited (red line) timing point
        # SampleSet: 1 = Normal, SampleIndex: 0 = default, Volume: 100, Uninherited: 1, Effects: 0
        line = f"{tp['offset_ms']},{tp['ms_per_beat']},{tp['meter']},1,0,100,1,0"
        lines.append(line)
    
    return "\n".join(lines)


def main():
    timing_profile = _timing_profile_from_argv(sys.argv[1:])
    defaults = _defaults_for_timing_profile(timing_profile)

    class _HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Convert audio to osu! timing points using Beat This! AI",
        formatter_class=_HelpFormatter,
        epilog=(
            "Timing profiles (recommended):\n"
            "  stable: metronomic / produced music (minimal redlines)\n"
            "  live:   live / rubato recordings (adds frequent resync redlines)\n"
            "  auto:   decide per-audio (gridfit vs resync)\n\n"
            "Examples:\n"
            "  python beat_to_osu.py song.mp3 -o timing_points.txt\n"
            "  python beat_to_osu.py song.mp3 --timing-profile live -o timing_points.txt\n"
        ),
    )
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--timing-profile",
        "--profile",
        dest="timing_profile",
        type=str,
        default=timing_profile,
        choices=sorted(TIMING_PROFILES.keys()),
        help="High-level preset for timing-point behavior. Advanced flags override these defaults.",
    )
    parser.add_argument(
        "--logit-refine",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("logit_refine", False)),
        help="Refine timing points by aligning each section to Beat This activations.",
    )
    parser.add_argument(
        "--onset-refine",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("onset_refine", False)),
        help="Refine timing points by aligning to audio onsets (spectral flux).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run inference on (cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--beat-this-checkpoint",
        "--checkpoint",
        dest="beat_this_checkpoint",
        type=str,
        default="final0",
        help="Beat This checkpoint shortname or path (e.g. final0, small0, or a local .ckpt).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=2.0,
        help="BPM change tolerance in milliseconds (default: 2.0)"
    )
    parser.add_argument(
        "--legacy-min-section-beats",
        type=int,
        default=4,
        help="Legacy mode: minimum IBIs before allowing a new timing point (default: %(default)s).",
    )
    parser.add_argument(
        "--legacy-mode",
        type=str,
        default=str(defaults.get("legacy_mode", "change")),
        choices=["change", "anchor"],
        help="Legacy mode: change-based segmentation vs anchor resync (default: %(default)s).",
    )
    parser.add_argument(
        "--legacy-max-section-beats",
        type=int,
        default=0,
        help="Legacy mode: force a new timing point every N IBIs (0 = disabled).",
    )
    parser.add_argument(
        "--legacy-max-section-ms",
        type=float,
        default=0.0,
        help="Legacy mode: force a new timing point after this many ms (0 = disabled).",
    )
    parser.add_argument(
        "--legacy-anchor-ms",
        type=float,
        default=float(defaults.get("legacy_anchor_ms", 4000.0)),
        help="Legacy anchor mode: resync interval in ms (default: %(default)s).",
    )
    parser.add_argument(
        "--legacy-anchor-alpha",
        type=float,
        default=float(defaults.get("legacy_anchor_alpha", 0.2)),
        help="Legacy anchor mode: local tempo adaptation alpha (default: %(default)s).",
    )
    parser.add_argument(
        "--legacy-aggressive",
        action="store_true",
        help="Legacy mode: paranoid resync defaults (1ms tolerance, 1-IBI min, resync every ~4s).",
    )
    parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("refine", False)),
        help="Refine beat times with sub-frame quadratic peak interpolation.",
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        default=str(defaults.get("postprocess", "minimal")),
        choices=["minimal", "viterbi", "dbn"],
        help="Beat postprocessing backend (default: %(default)s).",
    )
    parser.add_argument(
        "--beat-shift-ms",
        type=float,
        default=0.0,
        help="Shift all detected beat/downbeat times by this many ms (can be negative).",
    )
    parser.add_argument(
        "--viterbi-min-bpm",
        type=float,
        default=55.0,
        help="Viterbi mode: minimum BPM (default: %(default)s).",
    )
    parser.add_argument(
        "--viterbi-max-bpm",
        type=float,
        default=215.0,
        help="Viterbi mode: maximum BPM (default: %(default)s).",
    )
    parser.add_argument(
        "--viterbi-transition-lambda",
        type=float,
        default=100.0,
        help="Viterbi mode: tempo change penalty (default: %(default)s).",
    )
    parser.add_argument(
        "--viterbi-tempo-prior-lambda",
        type=float,
        default=30.0,
        help="Viterbi mode: prior strength toward the dominant tempo (default: %(default)s).",
    )
    parser.add_argument(
        "--viterbi-peak-threshold",
        type=float,
        default=0.2,
        help="Viterbi mode: minimum beat probability to consider a frame (default: %(default)s).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=str(defaults.get("method", "gridfit")),
        choices=["gridfit", "legacy", "elastic", "auto"],
        help="Timing point inference method (default: %(default)s).",
    )
    parser.add_argument(
        "--force-meter",
        type=int,
        default=None,
        help="Override detected meter (e.g., 3 or 4).",
    )
    parser.add_argument(
        "--auto-pass-ms",
        type=float,
        default=2.0,
        help="Auto mode: accept gridfit if mean residual <= this (ms).",
    )
    parser.add_argument(
        "--auto-fail-ms",
        type=float,
        default=5.0,
        help="Auto mode: switch to legacy if mean residual >= this (ms).",
    )
    parser.add_argument(
        "--auto-gray",
        type=str,
        default="gridfit",
        choices=["gridfit", "legacy"],
        help="Auto mode: method to use when residual is between pass/fail.",
    )
    parser.add_argument(
        "--hitobjects-json",
        type=str,
        default=None,
        help=(
            "Optional: path to a data/datasets2 annotation JSON (or similar) containing hit_objects[]. "
            "When provided, timing points are refined to better quantize these note times."
        ),
    )
    parser.add_argument(
        "--intent-offset-step-ms",
        type=float,
        default=1.0,
        help="Note-aware refinement: offset search step in ms (default: %(default)s).",
    )
    parser.add_argument(
        "--intent-mpb-search-pct",
        type=float,
        default=0.01,
        help="Note-aware refinement: search +/- pct around each section's ms_per_beat (default: %(default)s).",
    )
    parser.add_argument(
        "--intent-mpb-search-steps",
        type=int,
        default=7,
        help="Note-aware refinement: number of ms_per_beat candidates (default: %(default)s).",
    )
    parser.add_argument(
        "--intent-min-hits",
        type=int,
        default=8,
        help="Note-aware refinement: minimum hit objects per section to refine (default: %(default)s).",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Optional: write a debug bundle (raw logits, beats, metadata) to this directory.",
    )
    parser.add_argument(
        "--dump-spectrogram",
        action="store_true",
        help="Include a log-mel spectrogram in the debug bundle (can be large).",
    )
    parser.add_argument(
        "--elastic-min-section-beats",
        type=int,
        default=16,
        help="Elastic mode: minimum beats per section (default: %(default)s).",
    )
    parser.add_argument(
        "--elastic-segment-penalty-ms",
        type=float,
        default=250.0,
        help="Elastic mode: cost (ms) per new timing point (default: %(default)s).",
    )
    parser.add_argument(
        "--elastic-tempo-change-penalty",
        type=float,
        default=20.0,
        help="Elastic mode: penalty multiplier for BPM changes (default: %(default)s).",
    )
    parser.add_argument(
        "--elastic-max-sections",
        type=int,
        default=128,
        help="Elastic mode: cap number of timing points (default: %(default)s).",
    )
    parser.add_argument(
        "--elastic-dbn",
        action="store_true",
        help="Elastic mode: use DBN-constrained beat tracking (usually better for live).",
    )
    
    args = parser.parse_args()

    if args.elastic_dbn:
        # Backwards-compat flag: prefer DBN postprocessing if available.
        args.postprocess = "dbn"

    if args.legacy_aggressive:
        if args.tolerance == 2.0:
            args.tolerance = 1.0
        if args.legacy_min_section_beats == 4:
            args.legacy_min_section_beats = 1
        if args.legacy_mode == "change" and args.legacy_max_section_ms == 0.0:
            args.legacy_max_section_ms = 4000.0
        if args.legacy_mode == "change" and args.legacy_max_section_beats == 0:
            args.legacy_max_section_beats = 0
        if args.legacy_mode == "change":
            # Default aggressive behavior: anchor resync is more stable than
            # per-beat BPM jitter when Beat This beat picks are noisy.
            args.legacy_mode = "anchor"
            if args.legacy_anchor_ms == 4000.0:
                args.legacy_anchor_ms = 3000.0

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Processing: {audio_path}", file=sys.stderr)
    print(f"[INFO] Using device: {args.device}", file=sys.stderr)
    
    # Run inference + beat tracking.
    print("[INFO] Running Beat This! inference...", file=sys.stderr)
    selected_method = args.method

    refine_effective = bool(args.refine)
    if args.postprocess == "minimal" and args.method in {"gridfit", "auto"} and not args.refine:
        print("[INFO] minimal: enabling --refine for better BPM/offset", file=sys.stderr)
        refine_effective = True

    beat_times, downbeat_times, beat_logits, downbeat_logits = get_frames_and_beats(
        str(audio_path),
        device=args.device,
        refine=refine_effective,
        postprocess=str(args.postprocess),
        checkpoint_path=str(args.beat_this_checkpoint),
        viterbi_min_bpm=float(args.viterbi_min_bpm),
        viterbi_max_bpm=float(args.viterbi_max_bpm),
        viterbi_transition_lambda=float(args.viterbi_transition_lambda),
        viterbi_tempo_prior_lambda=float(args.viterbi_tempo_prior_lambda),
        viterbi_peak_threshold=float(args.viterbi_peak_threshold),
        meter_for_downbeats=int(args.force_meter) if args.force_meter is not None else 4,
    )

    if float(args.beat_shift_ms) != 0.0:
        shift_s = float(args.beat_shift_ms) / 1000.0
        beat_times = np.asarray(beat_times, dtype=np.float64) + shift_s
        downbeat_times = np.asarray(downbeat_times, dtype=np.float64) + shift_s

    beats, downbeats = beat_times.tolist(), downbeat_times.tolist()

    print(f"[INFO] Detected {len(beats)} beats and {len(downbeats)} downbeats", file=sys.stderr)
    
    # Convert to timing points
    if args.method == "legacy":
        if args.legacy_mode == "anchor":
            timing_points = beats_to_timing_points_anchor_resync(
                beats,
                downbeats,
                anchor_ms=float(args.legacy_anchor_ms),
                local_alpha=float(args.legacy_anchor_alpha),
                force_meter=int(args.force_meter) if args.force_meter is not None else 4,
            )
        else:
            timing_points = beats_to_timing_points(
                beats,
                downbeats,
                tolerance_ms=args.tolerance,
                min_section_beats=args.legacy_min_section_beats,
                max_section_beats=(
                    None
                    if int(args.legacy_max_section_beats) <= 0
                    else int(args.legacy_max_section_beats)
                ),
                max_section_ms=(
                    None if float(args.legacy_max_section_ms) <= 0 else float(args.legacy_max_section_ms)
                ),
                force_meter=int(args.force_meter) if args.force_meter is not None else None,
            )
    elif args.method == "elastic":
        timing_points = beats_to_timing_points_elastic(
            beats,
            downbeats,
            min_section_beats=args.elastic_min_section_beats,
            segment_penalty_ms=args.elastic_segment_penalty_ms,
            tempo_change_penalty=args.elastic_tempo_change_penalty,
            max_sections=args.elastic_max_sections,
            meter=int(args.force_meter) if args.force_meter is not None else 4,
        )
    else:
        offset_ms, mpb = _fit_timing_grid_from_logits(
            beat_logits=np.asarray(beat_logits, dtype=np.float64),
            beat_times_s=beat_times,
            fps=50,
            prefer_faster=True,
        )
        if args.method == "auto":
            residuals_ms = _gridfit_residuals_ms(
                beat_times_s=beat_times, ms_per_beat=mpb, phase_search_step_ms=1.0
            )
            if residuals_ms.size == 0:
                residual_mean_ms = float("inf")
                residual_median_ms = float("inf")
                residual_p95_ms = float("inf")
                residual_max_ms = float("inf")
            else:
                residual_mean_ms = float(np.mean(residuals_ms))
                residual_median_ms = float(np.median(residuals_ms))
                residual_p95_ms = float(np.percentile(residuals_ms, 95))
                residual_max_ms = float(np.max(residuals_ms))

            print(
                "[INFO] auto: gridfit residual mean=%.3fms median=%.3fms p95=%.3fms max=%.3fms"
                % (
                    residual_mean_ms,
                    residual_median_ms,
                    residual_p95_ms,
                    residual_max_ms,
                ),
                file=sys.stderr,
            )

            if residual_mean_ms <= args.auto_pass_ms:
                selected_method = "gridfit"
            elif residual_mean_ms >= args.auto_fail_ms:
                selected_method = "legacy"
            else:
                selected_method = args.auto_gray

            if selected_method == "legacy" and args.legacy_mode == "change":
                # For live music, prefer anchor resync over per-beat BPM jitter.
                args.legacy_mode = "anchor"

            print(
                f"[INFO] auto: decision={selected_method} (pass<= {args.auto_pass_ms}ms, "
                f"fail>= {args.auto_fail_ms}ms, gray={args.auto_gray})",
                file=sys.stderr,
            )

        if selected_method == "legacy":
            if args.legacy_mode == "anchor":
                timing_points = beats_to_timing_points_anchor_resync(
                    beats,
                    downbeats,
                    anchor_ms=float(args.legacy_anchor_ms),
                    local_alpha=float(args.legacy_anchor_alpha),
                    force_meter=int(args.force_meter) if args.force_meter is not None else 4,
                )
            else:
                timing_points = beats_to_timing_points(
                    beats,
                    downbeats,
                    tolerance_ms=args.tolerance,
                    min_section_beats=args.legacy_min_section_beats,
                    max_section_beats=(
                        None
                        if int(args.legacy_max_section_beats) <= 0
                        else int(args.legacy_max_section_beats)
                    ),
                    max_section_ms=(
                        None
                        if float(args.legacy_max_section_ms) <= 0
                        else float(args.legacy_max_section_ms)
                    ),
                    force_meter=int(args.force_meter) if args.force_meter is not None else None,
                )
        elif selected_method == "elastic":
            timing_points = beats_to_timing_points_elastic(
                beats,
                downbeats,
                min_section_beats=args.elastic_min_section_beats,
                segment_penalty_ms=args.elastic_segment_penalty_ms,
                tempo_change_penalty=args.elastic_tempo_change_penalty,
                max_sections=args.elastic_max_sections,
                meter=int(args.force_meter) if args.force_meter is not None else 4,
            )
        else:
            if args.force_meter is not None:
                meter = int(args.force_meter)
            else:
                meter = _meter_from_downbeats(
                    downbeats_s=downbeat_times, offset_ms=offset_ms, ms_per_beat=mpb
                )
            timing_points = [
                {
                    "offset_ms": round(offset_ms),
                    "ms_per_beat": round(mpb, 6),
                    "meter": meter,
                }
            ]

    per_section_logit_refine = None
    if bool(args.logit_refine) and len(timing_points) > 0:
        before_n = len(timing_points)
        timing_points, per_section_logit_refine = refine_timing_points_with_logits(
            timing_points,
            beat_logits=np.asarray(beat_logits, dtype=np.float64),
            fps=50.0,
        )
        print(
            f"[INFO] logit_refine: refined timing points using activations ({before_n} -> {len(timing_points)})",
            file=sys.stderr,
        )

    per_section_onset_refine = None
    if bool(args.onset_refine) and len(timing_points) > 0:
        onset_ms = detect_audio_onsets_ms(str(audio_path))
        if onset_ms.size:
            before_n = len(timing_points)
            timing_points, per_section_onset_refine = refine_timing_points_with_hitobjects(
                timing_points,
                onset_ms,
                offset_step_ms=1.0,
                max_offset_shift_ms=25.0,
                mpb_search_pct=0.01,
                mpb_search_steps=7,
                min_hits_per_section=32,
            )
            print(
                f"[INFO] onset_refine: refined timing points using {int(onset_ms.size)} onsets ({before_n} -> {len(timing_points)})",
                file=sys.stderr,
            )
    per_section_intent = None
    if args.hitobjects_json:
        hit_times_ms = _load_hit_objects_ms(str(args.hitobjects_json))
        before_n = len(timing_points)
        timing_points, per_section_intent = refine_timing_points_with_hitobjects(
            timing_points,
            hit_times_ms,
            offset_step_ms=float(args.intent_offset_step_ms),
            mpb_search_pct=float(args.intent_mpb_search_pct),
            mpb_search_steps=int(args.intent_mpb_search_steps),
            min_hits_per_section=int(args.intent_min_hits),
        )
        print(
            f"[INFO] intent: refined timing points using hitobjects ({before_n} -> {len(timing_points)})",
            file=sys.stderr,
        )

    print(f"[INFO] Generated {len(timing_points)} timing points", file=sys.stderr)
    
    if args.dump_dir:
        dump_meta = {
            "args": vars(args),
            "selected_method": selected_method,
            "auto_residual_mean_ms": residual_mean_ms if args.method == "auto" else None,
            "auto_residual_median_ms": residual_median_ms if args.method == "auto" else None,
            "auto_residual_p95_ms": residual_p95_ms if args.method == "auto" else None,
            "auto_residual_max_ms": residual_max_ms if args.method == "auto" else None,
            "logit_refine_sections": per_section_logit_refine,
            "onset_refine_sections": per_section_onset_refine,
            "intent_sections": per_section_intent,
        }
        dump_debug_bundle(
            Path(str(args.dump_dir)),
            audio_path=audio_path,
            fps=50,
            beat_times_s=np.asarray(beat_times, dtype=np.float64),
            downbeat_times_s=np.asarray(downbeat_times, dtype=np.float64),
            beat_logits=np.asarray(beat_logits, dtype=np.float64),
            downbeat_logits=np.asarray(downbeat_logits, dtype=np.float64),
            timing_points=timing_points,
            meta=dump_meta,
            dump_spectrogram=bool(args.dump_spectrogram),
        )
        print(f"[INFO] Wrote debug bundle to: {args.dump_dir}", file=sys.stderr)

    # Format output
    output_text = format_osu_timing_points(timing_points)
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"[INFO] Saved to: {output_path}", file=sys.stderr)
    else:
        print(output_text)
    
    # Also print BPM summary
    print("\n[SUMMARY]", file=sys.stderr)
    for i, tp in enumerate(timing_points):
        bpm = 60000.0 / tp['ms_per_beat']
        print(
            f"  Section {i+1}: {tp['offset_ms']}ms, {bpm:.2f} BPM, {tp['meter']}/4 time",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()
