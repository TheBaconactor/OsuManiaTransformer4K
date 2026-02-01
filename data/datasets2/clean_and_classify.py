"""
data/datasets2 cleaner + status classifier.

Goals:
  1) Detect basic dataset inconsistencies (missing audio for annotation, orphan audio).
  2) Classify each beatmapset into osu! rank status buckets (ranked/qualified/loved/pending/graveyard/wip/etc)
     using osu! API v2 when BeatmapSetID is available.
  3) Write gitignored index files under <dataset>/metadata/ for easier training/eval selection.

Notes:
  - Many locally-exported / synthetic maps may have BeatmapSetID=-1 or "unknown". Those are classified as "unknown".
  - This script does not delete anything by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_quotes(v)
        if k:
            out[k] = v
    return out


@dataclass(frozen=True)
class OsuAuth:
    client_id: str
    client_secret: str


def resolve_auth(env_file: Path | None, client_id: str | None, client_secret: str | None) -> OsuAuth:
    file_env: dict[str, str] = {}
    if env_file is not None:
        file_env = load_env_file(env_file)

    cid = client_id or os.environ.get("OSU_CLIENT_ID") or file_env.get("OSU_CLIENT_ID")
    csec = client_secret or os.environ.get("OSU_CLIENT_SECRET") or file_env.get("OSU_CLIENT_SECRET")

    if not cid or not csec:
        raise SystemExit(
            "Missing osu! API creds.\n"
            "Create config/osu_api.env with OSU_CLIENT_ID and OSU_CLIENT_SECRET (recommended),\n"
            "or pass --client-id/--client-secret."
        )
    return OsuAuth(client_id=cid, client_secret=csec)


def get_access_token(auth: OsuAuth) -> str:
    resp = requests.post(
        "https://osu.ppy.sh/oauth/token",
        data={
            "client_id": auth.client_id,
            "client_secret": auth.client_secret,
            "grant_type": "client_credentials",
            "scope": "public",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(f"Auth failed: HTTP {resp.status_code}: {resp.text}")
    token = resp.json().get("access_token")
    if not token:
        raise SystemExit("Auth failed: missing access_token in response")
    return token


def safe_int(value: Any) -> int | None:
    try:
        i = int(value)
    except Exception:
        return None
    return i


def is_valid_set_id(beatmapset_id: int | None) -> bool:
    return bool(beatmapset_id and beatmapset_id > 0)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def mkdir_clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    mkdir_clean(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    mkdir_clean(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def status_group(status: str) -> str:
    s = status.strip().lower()
    if s in {"ranked", "approved"}:
        return "ranked"
    if s in {"qualified"}:
        return "qualified"
    if s in {"loved"}:
        return "loved"
    if s in {"pending", "wip"}:
        return "pending"
    if s in {"graveyard"}:
        return "graveyard"
    if s in {"unknown"}:
        return "unknown"
    return "other"


def fetch_status_for_set(
    session: requests.Session,
    beatmapset_id: int,
    cache: dict[str, Any],
    rpm: int,
) -> dict[str, Any]:
    key = str(beatmapset_id)
    if key in cache:
        return cache[key]

    min_sleep = 60.0 / max(1, int(rpm))
    time.sleep(min_sleep)

    url = f"https://osu.ppy.sh/api/v2/beatmapsets/{beatmapset_id}"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        cache[key] = {"status": "unknown", "source": "api", "beatmapset_id": beatmapset_id}
        return cache[key]
    if resp.status_code != 200:
        cache[key] = {
            "status": "unknown",
            "source": "api_error",
            "beatmapset_id": beatmapset_id,
            "http_status": resp.status_code,
        }
        return cache[key]

    data = resp.json()
    status = (data.get("status") or "unknown").strip().lower()
    cache[key] = {
        "status": status,
        "source": "api",
        "beatmapset_id": beatmapset_id,
        "ranked_date": data.get("ranked_date"),
        "submitted_date": data.get("submitted_date"),
        "last_updated": data.get("last_updated"),
    }
    return cache[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and classify data/datasets2 by osu! rank status")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Dataset root folder (default: this script's directory)",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help="Where to write index files (default: <dataset-dir>/metadata)",
    )
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env (defaults to config/osu_api.env)")
    parser.add_argument("--client-id", type=str, default=None, help="osu! OAuth client id")
    parser.add_argument("--client-secret", type=str, default=None, help="osu! OAuth client secret")
    parser.add_argument("--rpm", type=int, default=60, help="Max osu! API requests per minute")
    parser.add_argument("--limit-sets", type=int, default=0, help="Limit API lookups (0 = no limit)")
    parser.add_argument("--apply", action="store_true", help="Apply cleanup actions (moves only)")
    parser.add_argument(
        "--quarantine-dir",
        type=str,
        default=None,
        help="When --apply, move orphans into this folder (default: <dataset-dir>/_quarantine)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    annotations_dir = dataset_dir / "annotations"
    audio_dir = dataset_dir / "audio"
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else (dataset_dir / "metadata")
    quarantine_dir = Path(args.quarantine_dir) if args.quarantine_dir else (dataset_dir / "_quarantine")

    mkdir_clean(metadata_dir)

    anno_paths = sorted(annotations_dir.glob("*.json"))
    audio_paths = sorted(audio_dir.glob("*.mp3"))

    items: list[dict[str, Any]] = []
    bad_json: list[str] = []

    for ap in anno_paths:
        data = load_json(ap)
        if data is None:
            bad_json.append(str(ap))
            continue
        items.append({"path": str(ap), "data": data})

    # Basic consistency checks
    referenced_audio: set[str] = set()
    missing_audio_for_annotation: list[str] = []
    for it in items:
        data = it["data"]
        audio_file = data.get("audio_file")
        if isinstance(audio_file, str):
            referenced_audio.add(audio_file)
            if not (audio_dir / audio_file).exists():
                missing_audio_for_annotation.append(it["path"])
        else:
            missing_audio_for_annotation.append(it["path"])

    orphan_audio: list[str] = []
    for p in audio_paths:
        if p.name not in referenced_audio:
            orphan_audio.append(str(p))

    # Load / init API cache
    repo_root = Path(__file__).resolve().parents[2]
    default_env = (repo_root / "config" / "osu_api.env").resolve()
    fallback_env = (repo_root / "data" / "osu2mir_audio" / ".env").resolve()
    if args.env_file:
        env_path = Path(args.env_file)
    else:
        env_path = default_env if default_env.exists() else fallback_env

    cache_path = metadata_dir / "beatmapset_status_cache.json"
    cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # Collect beatmapset IDs from annotations
    set_id_to_song_ids: dict[int, list[str]] = {}
    unknown_song_ids: list[str] = []
    for it in items:
        data = it["data"]
        song_id = data.get("song_id") or Path(it["path"]).stem
        bms_id = safe_int(data.get("beatmap_set_id"))
        if not is_valid_set_id(bms_id):
            unknown_song_ids.append(str(song_id))
            continue
        set_id_to_song_ids.setdefault(int(bms_id), []).append(str(song_id))

    # Fetch statuses
    set_status: dict[str, str] = {}  # beatmapset_id (str) -> status
    if set_id_to_song_ids:
        auth = resolve_auth(env_path, args.client_id, args.client_secret)
        token = get_access_token(auth)
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

        ids = sorted(set_id_to_song_ids.keys())
        if args.limit_sets and args.limit_sets > 0:
            ids = ids[: int(args.limit_sets)]

        for bms_id in ids:
            info = fetch_status_for_set(session, bms_id, cache, rpm=int(args.rpm))
            status = (info.get("status") or "unknown").strip().lower()
            set_status[str(bms_id)] = status

    # Build per-song status map
    song_status: dict[str, Any] = {}
    for it in items:
        data = it["data"]
        song_id = str(data.get("song_id") or Path(it["path"]).stem)
        bms_id = safe_int(data.get("beatmap_set_id"))
        if is_valid_set_id(bms_id):
            status = set_status.get(str(bms_id), cache.get(str(bms_id), {}).get("status", "unknown"))
        else:
            status = "unknown"
        song_status[song_id] = {
            "status": status,
            "group": status_group(status),
            "beatmap_set_id": bms_id,
            "beatmap_id": safe_int(data.get("beatmap_id")),
            "artist": data.get("artist"),
            "title": data.get("title"),
            "creator": data.get("creator"),
            "version": data.get("version"),
        }

    # Write outputs
    write_json(metadata_dir / "cleanup_report.json", {
        "annotations": len(anno_paths),
        "audio": len(audio_paths),
        "bad_json": bad_json,
        "missing_audio_for_annotation": missing_audio_for_annotation,
        "orphan_audio": orphan_audio,
    })
    write_json(cache_path, cache)
    write_json(metadata_dir / "song_status.json", song_status)

    # By-status lists
    buckets: dict[str, list[str]] = {}
    for song_id, info in song_status.items():
        buckets.setdefault(info["status"], []).append(song_id)

    by_status_dir = metadata_dir / "by_status"
    for status, song_ids in sorted(buckets.items(), key=lambda kv: kv[0]):
        write_text(by_status_dir / f"{status}.txt", "\n".join(sorted(song_ids)) + "\n")

    # Group lists
    groups: dict[str, list[str]] = {}
    for song_id, info in song_status.items():
        groups.setdefault(info["group"], []).append(song_id)
    by_group_dir = metadata_dir / "by_group"
    for grp, song_ids in sorted(groups.items(), key=lambda kv: kv[0]):
        write_text(by_group_dir / f"{grp}.txt", "\n".join(sorted(song_ids)) + "\n")

    # Optional cleanup actions (moves only; no deletes)
    if args.apply:
        mkdir_clean(quarantine_dir)
        if orphan_audio:
            qa = quarantine_dir / "orphan_audio"
            mkdir_clean(qa)
            for p in orphan_audio:
                src = Path(p)
                if src.exists():
                    src.replace(qa / src.name)
        if missing_audio_for_annotation:
            qm = quarantine_dir / "missing_audio_annotations"
            mkdir_clean(qm)
            for p in missing_audio_for_annotation:
                src = Path(p)
                if src.exists():
                    src.replace(qm / src.name)

    print(f"[OK] annotations={len(anno_paths)} audio={len(audio_paths)}")
    print(f"[OK] wrote: {metadata_dir / 'cleanup_report.json'}")
    print(f"[OK] wrote: {metadata_dir / 'song_status.json'}")
    print(f"[OK] wrote: {by_status_dir}/*.txt")
    print(f"[OK] wrote: {by_group_dir}/*.txt")
    if args.apply:
        print(f"[OK] quarantine: {quarantine_dir}")


if __name__ == "__main__":
    main()
