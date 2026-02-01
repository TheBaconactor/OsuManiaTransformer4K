# Dataset Gathering (osu! → `data/datasets2/` → Beat This fine-tune)

This repo supports two ways to build training/eval data:

1) **Local osu! exports** (`.osz` from the osu! client) → `data/datasets2/` (recommended)
2) **API search → ID list → mirror download** (`.osz`) → `data/datasets2/` (automated, but depends on mirrors)

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
2) Create `config/osu_api.env` from the template `config/osu_api.env.example`:

```env
OSU_CLIENT_ID=...
OSU_CLIENT_SECRET=...
```

`config/osu_api.env` is gitignored.

## 2) Fetch beatmapset IDs via the official API (filtered)

This uses the official v2 search endpoint to build `data/osu2mir_audio/beatmapset_ids.txt`:

```powershell
.\.venv\Scripts\python data\osu2mir_audio\fetch_beatmapset_ids.py `
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
.\.venv\Scripts\python data\datasets2\download_osz_by_ids.py `
  --ids-file data\osu2mir_audio\beatmapset_ids.txt `
  --out-dir data\datasets2\osz_cache `
  --limit 200 `
  --max-workers 3
```

This downloads `.osz` archives into `data/datasets2/osz_cache/` (gitignored).

## 4) Convert `.osz` → `data/datasets2/` (audio + annotations)

```powershell
.\.venv\Scripts\python data\datasets2\osz_to_dataset.py `
  --input-dir data\datasets2\osz_cache `
  --output-dir data\datasets2
```

Outputs go under:
- `data/datasets2/audio/`
- `data/datasets2/annotations/`
- `data/datasets2/timing_points/`
- `data/datasets2/metadata/`

All are gitignored.

## 4.5) Classify and index by ranked status (optional but recommended)

This uses osu! API v2 to classify beatmapsets (ranked/qualified/loved/pending/graveyard/...) and writes index files under `data/datasets2/metadata/`:

```powershell
.\.venv\Scripts\python data\datasets2\clean_and_classify.py --rpm 60
```

Outputs include:
- `data/datasets2/metadata/song_status.json` (per-song status info)
- `data/datasets2/metadata/by_status/*.txt` (song_id lists)
- `data/datasets2/metadata/by_group/*.txt` (ranked/qualified/loved/pending/graveyard/unknown buckets)

## 5) Build Beat This training data from `data/datasets2/`

```powershell
python training\scripts\build_beat_this_osu_dataset.py
```

Optional smoke test:

```powershell
python training\scripts\build_beat_this_osu_dataset.py --max-items 1 --no-spectrograms
```

## Notes / caveats

- Mirror downloading is not the official osu! file distribution mechanism. Use responsibly, respect rate limits, and prefer local exports when possible.
- If you already have `.osz` exports from the osu! client, you can skip steps 2–3 and just run:
  - `python data\datasets2\osz_to_dataset.py` (reads from `C:\Users\troll\AppData\Local\osu!\Exports` by default).
