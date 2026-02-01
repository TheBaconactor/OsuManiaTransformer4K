# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `data/osu2mir_audio/` and below.
- **Key scripts:**
  - download_beatmaps.py (osu! API v2 downloader)
  - fetch_beatmapset_ids.py (osu! API v2 search -> ID lists)

## Commands
- **Fetch IDs:** `python data/osu2mir_audio/fetch_beatmapset_ids.py --status ranked --mode mania --keys 4 --max-sets 200`
- **Download beatmaps:** `python data/osu2mir_audio/download_beatmaps.py --client-id <id> --client-secret <secret>`

## Conventions
- Requires osu! API v2 credentials (OAuth client id/secret).
- Outputs go to `data/osu2mir_audio/audio`, `data/osu2mir_audio/annotations`, `data/osu2mir_audio/osz_cache` (gitignored).
- Keep secrets out of code and shell history.

## Do not
- Commit downloaded audio, annotations, or API credentials.
