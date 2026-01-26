"""
Beat This! to Osu! Timing Point Converter

This script uses the SOTA 'Beat This!' model to detect beats in an audio file
and converts them into osu! .osu format timing points.

Usage:
    python beat_to_osu.py path/to/audio.mp3
    python beat_to_osu.py path/to/audio.mp3 --output timing_points.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np


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

        a2f = OnnxAudio2Frames(
            checkpoint_path="final0",
            provider="DmlExecutionProvider",
            float16=False,
        )
        beat_logits_t, downbeat_logits_t = a2f(signal, sr)
    else:
        from beat_this.inference import Audio2Frames

        a2f = Audio2Frames(
            checkpoint_path="final0",
            device=resolved_device,
            float16=False,
        )
        beat_logits_t, downbeat_logits_t = a2f(signal, sr)

    if postprocess not in {"minimal", "dbn", "viterbi"}:
        raise ValueError(f"Unknown postprocess={postprocess!r}")

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
        beat_time = beat_frames.astype(np.float64) / 50.0
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
                beat_time = beat_frames.astype(np.float64) / 50.0
                downbeat_time = downbeat_frames.astype(np.float64) / 50.0
            else:
                raise
        else:
            beat_time, downbeat_time = post(beat_logits_t, downbeat_logits_t)

    if refine:
        # Refinement only applies to peak-picked beats.
        if postprocess == "minimal":
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
    audio_path: str, device: str = "cpu", refine: bool = False, dbn: bool = False
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
                checkpoint_path="final0",
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
            checkpoint_path="final0",
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
                checkpoint_path="final0",
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
    parser = argparse.ArgumentParser(
        description="Convert audio to osu! timing points using Beat This! AI"
    )
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run inference on (cpu, cuda, cuda:0, etc.)"
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
        default="change",
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
        default=4000.0,
        help="Legacy anchor mode: resync interval in ms (default: %(default)s).",
    )
    parser.add_argument(
        "--legacy-anchor-alpha",
        type=float,
        default=0.2,
        help="Legacy anchor mode: local tempo adaptation alpha (default: %(default)s).",
    )
    parser.add_argument(
        "--legacy-aggressive",
        action="store_true",
        help="Legacy mode: paranoid resync defaults (1ms tolerance, 1-IBI min, resync every ~4s).",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine beat times with sub-frame quadratic peak interpolation.",
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        default="minimal",
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
        default="gridfit",
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

    beat_times, downbeat_times, beat_logits, _downbeat_logits = get_frames_and_beats(
        str(audio_path),
        device=args.device,
        refine=refine_effective,
        postprocess=str(args.postprocess),
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
    print(f"[INFO] Generated {len(timing_points)} timing points", file=sys.stderr)
    
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
