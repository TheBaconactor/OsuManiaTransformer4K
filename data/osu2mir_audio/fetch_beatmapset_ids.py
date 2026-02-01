"""
Fetch beatmapset IDs via the official osu! API v2 search endpoint.

This is intended to build a filtered list of BeatmapSetIDs for downloading .osz
archives (for Dataset2 / training). It does NOT download the .osz itself.

Credentials:
  - Put OSU_CLIENT_ID / OSU_CLIENT_SECRET in config/osu_api.env (gitignored), or
  - Provide --client-id/--client-secret, or
  - Provide OSU_CLIENT_ID / OSU_CLIENT_SECRET in the environment.

Example:
  python data/osu2mir_audio/fetch_beatmapset_ids.py --status ranked qualified --mode mania --keys 4 --max-sets 200
"""

from __future__ import annotations

import argparse
import os
import sys
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
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise SystemExit("Auth failed: missing access_token in response")
    return token


def mode_to_int(mode: str) -> int:
    m = mode.strip().lower()
    if m in ("0", "osu", "std", "standard"):
        return 0
    if m in ("1", "taiko"):
        return 1
    if m in ("2", "catch", "ctb", "fruits"):
        return 2
    if m in ("3", "mania"):
        return 3
    raise SystemExit(f"Unknown mode: {mode!r}")


def beatmap_is_keys_compatible(beatmap: dict[str, Any], mode: int, keys: int | None) -> bool:
    if mode == 3:
        # API typically exposes beatmap["mode"] as a string and keycount via beatmap["cs"].
        bm_mode = beatmap.get("mode")
        if bm_mode not in (None, "mania"):
            return False
        if keys is None:
            return True
        cs = beatmap.get("cs")
        try:
            return int(round(float(cs))) == int(keys)
        except Exception:
            return False
    # For non-mania we don't key-filter.
    return True


def fetch_ids(
    token: str,
    query: str,
    status: str,
    mode: int,
    keys: int | None,
    max_sets: int,
    rpm: int,
) -> list[int]:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    ids: list[int] = []
    cursor_string: str | None = None
    min_sleep = 60.0 / max(1, rpm)

    while len(ids) < max_sets:
        params: dict[str, Any] = {"q": query, "m": mode, "s": status}
        if cursor_string:
            params["cursor_string"] = cursor_string

        resp = session.get("https://osu.ppy.sh/api/v2/beatmapsets/search", params=params, timeout=30)
        if resp.status_code != 200:
            raise SystemExit(f"Search failed: HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        beatmapsets = data.get("beatmapsets") or []
        if not beatmapsets:
            break

        for bms in beatmapsets:
            if len(ids) >= max_sets:
                break
            bms_id = bms.get("id")
            if bms_id is None:
                continue

            # Optional: require at least one beatmap matching mode/keys.
            if keys is not None and mode == 3:
                beatmaps = bms.get("beatmaps") or []
                if not any(beatmap_is_keys_compatible(bm, mode=mode, keys=keys) for bm in beatmaps):
                    continue

            try:
                ids.append(int(bms_id))
            except Exception:
                continue

        cursor_string = (data.get("cursor_string") or "").strip() or None
        if not cursor_string:
            break

        time.sleep(min_sleep)

    # De-dupe while preserving order
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:max_sets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch beatmapset IDs via osu! API v2 search")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env (defaults to config/osu_api.env)")
    parser.add_argument("--client-id", type=str, default=None, help="osu! OAuth client id")
    parser.add_argument("--client-secret", type=str, default=None, help="osu! OAuth client secret")
    parser.add_argument("--query", type=str, default="", help="Search query (q)")
    parser.add_argument(
        "--status",
        type=str,
        nargs="+",
        default=["ranked"],
        help="Search status(es): ranked|qualified|loved|graveyard|... (s). Can pass multiple.",
    )
    parser.add_argument("--mode", type=str, default="mania", help="Mode: osu|taiko|catch|mania (m)")
    parser.add_argument("--keys", type=int, default=4, help="Key count filter for mania (uses beatmap cs)")
    parser.add_argument("--max-sets", type=int, default=200, help="Maximum number of beatmapset IDs to output")
    parser.add_argument("--rpm", type=int, default=60, help="Max requests per minute (default 60)")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "beatmapset_ids.txt"),
        help="Output file path (one beatmapset id per line)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_root = script_dir.parents[2]
    if args.env_file:
        env_path = Path(args.env_file)
    else:
        preferred_env = repo_root / "config" / "osu_api.env"
        fallback_env = script_dir / ".env"
        env_path = preferred_env if preferred_env.exists() else fallback_env

    auth = resolve_auth(env_path, args.client_id, args.client_secret)
    token = get_access_token(auth)

    mode = mode_to_int(args.mode)
    keys = args.keys if (mode == 3 and args.keys and args.keys > 0) else None

    statuses = [str(s).strip() for s in (args.status or []) if str(s).strip()]
    if not statuses:
        raise SystemExit("No --status provided")

    ids_by_status: dict[str, list[int]] = {}
    all_ids: list[int] = []
    for status in statuses:
        ids = fetch_ids(
            token=token,
            query=args.query,
            status=status,
            mode=mode,
            keys=keys,
            max_sets=max(1, args.max_sets),
            rpm=max(1, args.rpm),
        )
        ids_by_status[status] = ids
        all_ids.extend(ids)

    # De-dupe combined list while preserving order
    seen: set[int] = set()
    combined: list[int] = []
    for i in all_ids:
        if i not in seen:
            seen.add(i)
            combined.append(i)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(i) for i in combined) + ("\n" if combined else ""), encoding="utf-8")
    print(f"[OK] Wrote {len(combined)} beatmapset IDs to {out_path}")

    if len(statuses) > 1:
        for status, ids in ids_by_status.items():
            status_path = out_path.with_name(f"{out_path.stem}.{status}{out_path.suffix}")
            status_path.write_text("\n".join(str(i) for i in ids) + ("\n" if ids else ""), encoding="utf-8")
            print(f"[OK] Wrote {len(ids)} ({status}) to {status_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Cancelled", file=sys.stderr)
        raise SystemExit(130)
