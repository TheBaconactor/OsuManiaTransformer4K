"""
Build a Beat This!-style training dataset from this repo's `data/datasets2/`.

Why:
  - We want to improve "The Ear" (timing) with *automatic supervision*.
  - Ranked maps already contain mapper timing points (redlines) which define a beat grid.
  - Beat This!'s training code expects:
      data/
        audio/spectrograms/<dataset>/<piece>/track.npy
        annotations/<dataset>/
          info.json
          single.split
          annotations/beats/<piece>.beats

This script converts `data/datasets2/annotations/*.json` + `data/datasets2/audio/*.mp3` into
that structure under `modules/beat_this/data/` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
BEAT_THIS_ROOT = REPO_ROOT / "modules" / "beat_this"
if str(BEAT_THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(BEAT_THIS_ROOT))

from beat_this.preprocessing import LogMelSpect, load_audio  # noqa: E402


@dataclass(frozen=True)
class _Section:
    start_ms: float
    mpb_ms: float
    meter: int


def _iter_redline_sections(timing_points: list[dict], audio_len_ms: float) -> list[_Section]:
    red = [
        tp
        for tp in timing_points
        if bool(tp.get("uninherited")) and float(tp.get("beat_length", 0.0)) > 0.0
    ]
    red.sort(key=lambda tp: float(tp.get("offset", 0.0)))
    if not red:
        return []

    sections: list[_Section] = []
    for i, tp in enumerate(red):
        start_ms = float(tp.get("offset", 0.0))
        mpb_ms = float(tp.get("beat_length", 0.0))
        meter = int(tp.get("meter", 4))
        if mpb_ms <= 0 or not np.isfinite(mpb_ms):
            continue
        # Clamp insane tempos (safety guard).
        if mpb_ms < 40.0:  # >1500 BPM
            continue
        end_ms = float(red[i + 1].get("offset", audio_len_ms)) if i + 1 < len(red) else float(audio_len_ms)
        if not np.isfinite(start_ms) or not np.isfinite(end_ms) or end_ms <= start_ms:
            continue
        sections.append(_Section(start_ms=max(0.0, start_ms), mpb_ms=mpb_ms, meter=max(1, meter)))
    return sections


def _beats_from_sections(sections: list[_Section], audio_len_ms: float) -> tuple[np.ndarray, np.ndarray]:
    if not sections:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int32)

    beats_ms: list[float] = []
    beat_numbers: list[int] = []
    for i, sec in enumerate(sections):
        start_ms = float(sec.start_ms)
        mpb_ms = float(sec.mpb_ms)
        meter = int(sec.meter)
        end_ms = float(sections[i + 1].start_ms) if i + 1 < len(sections) else float(audio_len_ms)
        if end_ms <= start_ms:
            continue
        # Generate beats for this segment. Count within each segment (beat 1 = downbeat).
        t = start_ms
        j = 0
        # Avoid runaway in case of corrupted inputs.
        max_beats = int(max(8, np.ceil((end_ms - start_ms) / max(mpb_ms, 1.0)) + 8))
        while t < end_ms and j < max_beats:
            beats_ms.append(float(t))
            beat_numbers.append(int((j % meter) + 1))
            j += 1
            t = start_ms + j * mpb_ms

    if not beats_ms:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int32)

    beats_ms_arr = np.asarray(beats_ms, dtype=np.float64)
    beat_numbers_arr = np.asarray(beat_numbers, dtype=np.int32)

    # Enforce strict monotonicity and de-dup.
    order = np.argsort(beats_ms_arr, kind="stable")
    beats_ms_arr = beats_ms_arr[order]
    beat_numbers_arr = beat_numbers_arr[order]
    keep = np.ones(beats_ms_arr.size, dtype=bool)
    keep[1:] = beats_ms_arr[1:] > beats_ms_arr[:-1]
    beats_ms_arr = beats_ms_arr[keep]
    beat_numbers_arr = beat_numbers_arr[keep]

    # Clip to audio duration.
    beats_ms_arr = beats_ms_arr[(beats_ms_arr >= 0.0) & (beats_ms_arr <= float(audio_len_ms))]
    beat_numbers_arr = beat_numbers_arr[: beats_ms_arr.size]

    return beats_ms_arr, beat_numbers_arr


def _audio_duration_ms(path: Path, target_sr: int) -> tuple[np.ndarray, int, float]:
    # We load audio anyway to build the spectrogram; return waveform (mono, float32), sr, duration_ms.
    signal, sr = load_audio(str(path), dtype="float32")
    if signal.ndim == 2:
        # load_audio uses channels_first=False, so (T, C)
        signal = signal.mean(axis=1)
    if signal.ndim != 1:
        raise RuntimeError(f"Unexpected audio shape {signal.shape} for {path}")

    waveform = torch.as_tensor(signal, dtype=torch.float32)
    if int(sr) != int(target_sr):
        waveform = torchaudio.functional.resample(waveform, orig_freq=int(sr), new_freq=int(target_sr))
        sr = int(target_sr)
    duration_ms = 1000.0 * float(waveform.numel()) / float(sr)
    return waveform, int(sr), duration_ms


def _write_beats_file(path: Path, beats_ms: np.ndarray, beat_numbers: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    beats_s = beats_ms.astype(np.float64) / 1000.0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for t, n in zip(beats_s.tolist(), beat_numbers.tolist()):
            f.write(f"{t:.6f}\t{int(n)}\n")


def _save_spectrogram(path: Path, spect: torch.Tensor, *, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "float16":
        arr = spect.detach().cpu().numpy().astype(np.float16, copy=False)
    elif dtype == "float32":
        arr = spect.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown spect dtype: {dtype!r}")
    np.save(str(path), arr)


def _load_datasets2_items(annotations_dir: Path) -> list[Path]:
    return sorted(annotations_dir.glob("*.json"))


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _load_song_status_map(datasets2_root: Path) -> dict[str, dict]:
    """
    Optional helper: load `<datasets2>/metadata/song_status.json`.
    Produced by `data/datasets2/clean_and_classify.py`.
    """
    path = datasets2_root / "metadata" / "song_status.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _item_group(song_status: dict[str, dict], song_id: str) -> str | None:
    info = song_status.get(song_id)
    if not isinstance(info, dict):
        return None
    grp = info.get("group")
    return str(grp) if isinstance(grp, str) else None


def _item_status(song_status: dict[str, dict], song_id: str) -> str | None:
    info = song_status.get(song_id)
    if not isinstance(info, dict):
        return None
    st = info.get("status")
    return str(st) if isinstance(st, str) else None


def _choose_representative(group_items: list[dict]) -> dict:
    """
    Pick one representative difficulty among duplicates (same beatmapset/audio).
    This avoids overweighting a single song via multiple difficulties.
    """

    def key(it: dict) -> tuple[int, int, str]:
        return (
            int(it.get("hit_objects_count") or 0),
            int(it.get("timing_points_count") or 0),
            str(it.get("song_id") or ""),
        )

    return sorted(group_items, key=key, reverse=True)[0]


def _split_train_val(
    items: list[dict],
    *,
    seed: int,
    val_ratio: float,
    val_artist: str | None,
) -> tuple[list[dict], list[dict]]:
    if not items:
        return [], []

    if len(items) == 1:
        # Useful for quick smoke tests (e.g., --max-items 1).
        return list(items), list(items)

    if val_artist:
        artist_norm = val_artist.casefold().strip()
        val = [it for it in items if str(it.get("artist", "")).casefold().strip() == artist_norm]
        train = [it for it in items if it not in val]
        if val and train:
            return train, val
        # Fall back to ratio split if filter is degenerate.

    rng = random.Random(int(seed))
    items_shuffled = list(items)
    rng.shuffle(items_shuffled)
    n_val = int(round(float(val_ratio) * len(items_shuffled)))
    n_val = max(1, min(n_val, len(items_shuffled) - 1))
    val = items_shuffled[:n_val]
    train = items_shuffled[n_val:]
    return train, val


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Beat This dataset from data/datasets2.")
    parser.add_argument(
        "--datasets2",
        type=str,
        default=str(REPO_ROOT / "data" / "datasets2"),
        help="Path to data/datasets2 folder (contains audio/ and annotations/).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(BEAT_THIS_ROOT / "data"),
        help="Output Beat This data dir (default: modules/beat_this/data).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="osu_datasets2",
        help="Name of the dataset folder under data/annotations and data/audio/spectrograms.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Resample audio to this rate before spectrogram (Beat This default).",
    )
    parser.add_argument(
        "--no-spectrograms",
        action="store_true",
        help="Only write .beats + split files (skip computing spectrograms).",
    )
    parser.add_argument(
        "--spect-dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Spectrogram dtype on disk. Use float32 for CPU training; float16 saves space (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if files already exist.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="If >0, only process the first N annotation files (for quick tests).",
    )
    parser.add_argument(
        "--allow-groups",
        type=str,
        default="",
        help="Comma-separated groups to keep from <datasets2>/metadata/song_status.json (e.g. ranked,qualified,loved). Empty = keep all.",
    )
    parser.add_argument(
        "--allow-status",
        type=str,
        default="",
        help="Comma-separated raw statuses to keep (e.g. ranked,qualified,loved). Empty = keep all.",
    )
    parser.add_argument(
        "--dedupe-by",
        type=str,
        default="none",
        choices=["none", "beatmapset", "audio"],
        help="Prevent duplicates by keeping only one difficulty per beatmapset or per audio file (default: %(default)s).",
    )
    parser.add_argument(
        "--val-artist",
        type=str,
        default="",
        help="If set, put all tracks whose annotation artist matches into validation.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio (used when --val-artist isn't usable).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Split RNG seed.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    datasets2 = Path(args.datasets2)
    ann_dir = datasets2 / "annotations"
    audio_dir = datasets2 / "audio"
    if not ann_dir.exists() or not audio_dir.exists():
        raise RuntimeError(f"Expected {ann_dir} and {audio_dir} to exist.")

    out_dir = Path(args.out)
    dataset_name = str(args.dataset_name).strip()
    if not dataset_name:
        raise ValueError("--dataset-name must be non-empty.")

    out_ann_dir = out_dir / "annotations" / dataset_name
    out_beats_dir = out_ann_dir / "annotations" / "beats"
    out_spects_dir = out_dir / "audio" / "spectrograms" / dataset_name

    ann_paths = _load_datasets2_items(ann_dir)
    if int(args.max_items) > 0:
        ann_paths = ann_paths[: int(args.max_items)]

    items: list[dict] = []
    for p in ann_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        song_id = str(data.get("song_id", "")).strip()
        audio_file = str(data.get("audio_file", "")).strip()
        if not song_id or not audio_file:
            continue
        audio_path = audio_dir / audio_file
        if not audio_path.exists():
            continue
        timing_points = data.get("timing_points", [])
        items.append(
            {
                "song_id": song_id,
                "audio_path": audio_path,
                "audio_file": audio_file,
                "timing_points": timing_points,
                "timing_points_count": len(timing_points) if isinstance(timing_points, list) else 0,
                "artist": data.get("artist", ""),
                "beatmap_set_id": _safe_int(data.get("beatmap_set_id")),
                "beatmap_id": _safe_int(data.get("beatmap_id")),
                "hit_objects_count": len(data.get("hit_objects", []) or []),
            }
        )

    # Optional: filter using dataset classification output to reduce poisoned labels.
    song_status = _load_song_status_map(datasets2)
    allow_groups = {s.strip().lower() for s in str(args.allow_groups).split(",") if s.strip()}
    allow_status = {s.strip().lower() for s in str(args.allow_status).split(",") if s.strip()}
    if allow_groups or allow_status:
        kept: list[dict] = []
        dropped = 0
        for it in items:
            sid = str(it.get("song_id") or "")
            grp = (_item_group(song_status, sid) or "").strip().lower()
            st = (_item_status(song_status, sid) or "").strip().lower()
            if allow_groups and grp and grp not in allow_groups:
                dropped += 1
                continue
            if allow_status and st and st not in allow_status:
                dropped += 1
                continue
            kept.append(it)
        items = kept
        if dropped:
            print(f"[INFO] Filtered out {dropped} items by status/group.")

    # Optional: de-duplicate by beatmapset_id or audio_file.
    dedupe_by: Literal["none", "beatmapset", "audio"] = str(args.dedupe_by)
    if dedupe_by != "none":
        buckets: dict[str, list[dict]] = {}
        for it in items:
            if dedupe_by == "beatmapset":
                bms_id = it.get("beatmap_set_id")
                key = f"bms:{int(bms_id)}" if isinstance(bms_id, int) and bms_id > 0 else f"unknown:{it['song_id']}"
            else:
                af = it.get("audio_file")
                key = f"audio:{af}" if isinstance(af, str) and af else f"unknown:{it['song_id']}"
            buckets.setdefault(key, []).append(it)

        deduped: list[dict] = []
        for group_items in buckets.values():
            if len(group_items) == 1:
                deduped.append(group_items[0])
            else:
                deduped.append(_choose_representative(group_items))
        if len(deduped) != len(items):
            print(f"[INFO] Dedupe ({dedupe_by}): {len(items)} -> {len(deduped)}")
        items = sorted(deduped, key=lambda it: str(it.get("song_id") or ""))

    if not items:
        print("[ERROR] No valid items found in data/datasets2.", file=sys.stderr)
        return 2

    # Prepare split.
    train_items, val_items = _split_train_val(
        items,
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        val_artist=(str(args.val_artist).strip() or None),
    )

    # Write info.json + split file.
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    (out_ann_dir / "info.json").write_text(
        json.dumps({"has_downbeats": True}, indent=2) + "\n", encoding="utf-8"
    )
    with (out_ann_dir / "single.split").open("w", encoding="utf-8", newline="\n") as f:
        for it in sorted(train_items, key=lambda x: x["song_id"]):
            f.write(f"{it['song_id']}\ttrain\n")
        for it in sorted(val_items, key=lambda x: x["song_id"]):
            f.write(f"{it['song_id']}\tval\n")

    # Prepare spectrogram transform once.
    mel = LogMelSpect(sample_rate=int(args.sample_rate), hop_length=441, n_fft=1024)

    written_beats = 0
    written_spects = 0
    skipped = 0
    for it in items:
        song_id = it["song_id"]
        audio_path = Path(it["audio_path"])
        beats_path = out_beats_dir / f"{song_id}.beats"
        spect_path = out_spects_dir / song_id / "track.npy"

        if not bool(args.overwrite) and beats_path.exists() and (bool(args.no_spectrograms) or spect_path.exists()):
            skipped += 1
            continue

        waveform, sr, duration_ms = _audio_duration_ms(audio_path, int(args.sample_rate))
        sections = _iter_redline_sections(it["timing_points"], duration_ms)
        beats_ms, beat_numbers = _beats_from_sections(sections, duration_ms)
        if beats_ms.size < 2:
            print(f"[WARN] No usable beats for {song_id}; skipping.", file=sys.stderr)
            continue

        _write_beats_file(beats_path, beats_ms, beat_numbers)
        written_beats += 1

        if not bool(args.no_spectrograms):
            with torch.no_grad():
                spect = mel(waveform)
            _save_spectrogram(spect_path, spect, dtype=str(args.spect_dtype))
            written_spects += 1

    print(f"[OK] Dataset: {dataset_name}")
    print(f"  items={len(items)} train={len(train_items)} val={len(val_items)}")
    print(f"  wrote beats={written_beats} spects={written_spects} skipped={skipped}")
    print(f"  output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
