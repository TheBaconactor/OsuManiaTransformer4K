"""
Download .osz archives for a list of BeatmapSetIDs.

This uses community mirrors because the official osu! API v2 does not provide
direct download links for beatmap archives.

Output goes to data/datasets2/osz_cache (gitignored).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests


def iter_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        try:
            ids.append(int(t))
        except ValueError:
            continue
    # de-dupe
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def download_one(beatmapset_id: int, out_dir: Path, session: requests.Session, timeout_s: int) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{beatmapset_id}.osz"
    if out_path.exists():
        return out_path

    urls = [
        f"https://catboy.best/d/{beatmapset_id}",
        f"https://api.chimu.moe/v1/download/{beatmapset_id}",
        f"https://osu.ppy.sh/beatmapsets/{beatmapset_id}/download",
    ]

    for url in urls:
        try:
            r = session.get(url, stream=True, timeout=timeout_s)
            if r.status_code != 200:
                continue
            tmp = out_path.with_suffix(".osz.part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            tmp.replace(out_path)
            return out_path
        except Exception:
            continue
    return None


def chunks(xs: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download .osz archives by beatmapset ID list")
    parser.add_argument("--ids-file", type=str, required=True, help="Text file with BeatmapSetIDs (one per line)")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "osz_cache"), help="Output dir")
    parser.add_argument("--limit", type=int, default=200, help="Max ids to download")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel downloads")
    parser.add_argument("--timeout", type=int, default=90, help="Per-request timeout (seconds)")
    parser.add_argument("--rpm", type=int, default=30, help="Soft rate limit for mirror requests (requests/minute)")
    args = parser.parse_args()

    ids = iter_ids(Path(args.ids_file))[: max(0, args.limit)]
    if not ids:
        raise SystemExit("No IDs found to download")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "OsuManiaTransformer4K-dataset/1.0"})
    min_sleep = 60.0 / max(1, int(args.rpm))

    ok = 0
    fail = 0

    # Throttle by chunking: each chunk launches up to max_workers then sleeps.
    for batch in chunks(ids, max(1, int(args.max_workers))):
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
            for bid in batch:
                futures.append(ex.submit(download_one, bid, out_dir, session, int(args.timeout)))
            for fut in as_completed(futures):
                p = fut.result()
                if p is None:
                    fail += 1
                else:
                    ok += 1
        time.sleep(min_sleep)

    print(f"[OK] Downloaded: {ok}")
    print(f"[FAIL] Failed: {fail}")
    print(f"[INFO] Cache dir: {out_dir}")


if __name__ == "__main__":
    main()
