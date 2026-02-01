# Dataset Gathering (osu! → Datasets2 → Beat This fine-tune)

This repo supports two ways to build training/eval data:

1) **Local osu! exports** (`.osz` from the osu! client) → `Datasets2/` (recommended)
2) **API search → ID list → mirror download** (`.osz`) → `Datasets2/` (automated, but depends on mirrors)

All downloaded/processed datasets are **gitignored** (do not commit audio, `.osz`, or generated annotations).

## 0) Create a Python env (Windows)

From repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install requests
```

## 1) Set up osu! API credentials (file-based)

1) Create an OAuth app on your osu! account page (OAuth section).
   - Callback URL is required by osu!, but for our **client credentials** flow it can be a dummy value like `http://localhost`.
2) Create `Datasets/.env` from the template `Datasets/.env.example`:

```env
OSU_CLIENT_ID=...
OSU_CLIENT_SECRET=...
```

`Datasets/.env` is gitignored.

## 2) Fetch beatmapset IDs via the official API (filtered)

This uses the official v2 search endpoint to build `Datasets/beatmapset_ids.txt`:

```powershell
.\.venv\Scripts\python Datasets\fetch_beatmapset_ids.py `
  --status ranked `
  --mode mania `
  --keys 4 `
  --max-sets 200
```

Notes:
- This step uses your creds to obtain an access token, then performs API search requests.
- The output ID list file is gitignored.

## 3) Download `.osz` for those IDs (mirror-based)

```powershell
.\.venv\Scripts\python Datasets2\download_osz_by_ids.py `
  --ids-file Datasets\beatmapset_ids.txt `
  --out-dir Datasets2\osz_cache `
  --limit 200 `
  --max-workers 3
```

This downloads `.osz` archives into `Datasets2/osz_cache/` (gitignored).

## 4) Convert `.osz` → `Datasets2/` (audio + annotations)

```powershell
.\.venv\Scripts\python Datasets2\osz_to_dataset.py `
  --input-dir Datasets2\osz_cache `
  --output-dir Datasets2
```

Outputs go under:
- `Datasets2/audio/`
- `Datasets2/annotations/`
- `Datasets2/timing_points/`
- `Datasets2/metadata/`

All are gitignored.

## 5) Build Beat This training data from `Datasets2/`

```powershell
python scripts\build_beat_this_osu_dataset.py
```

Optional smoke test:

```powershell
python scripts\build_beat_this_osu_dataset.py --max-items 1 --no-spectrograms
```

## Notes / caveats

- Mirror downloading is not the official osu! file distribution mechanism. Use responsibly, respect rate limits, and prefer local exports when possible.
- If you already have `.osz` exports from the osu! client, you can skip steps 2–3 and just run:
  - `python Datasets2\osz_to_dataset.py` (reads from `C:\Users\troll\AppData\Local\osu!\Exports` by default).

