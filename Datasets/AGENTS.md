# Agent instructions (scope: this directory and subdirectories)

## Scope and layout
- **This AGENTS.md applies to:** `Datasets/` and below.
- **Key scripts:**
  - download_beatmaps.py (osu! API v2 downloader)

## Commands
- **Download beatmaps:** `python Datasets/download_beatmaps.py --client-id <id> --client-secret <secret>`

## Conventions
- Requires osu! API v2 credentials (OAuth client id/secret).
- Outputs go to `Datasets/audio`, `Datasets/annotations`, `Datasets/osz_cache` (gitignored).
- Keep secrets out of code and shell history.

## Do not
- Commit downloaded audio, annotations, or API credentials.
