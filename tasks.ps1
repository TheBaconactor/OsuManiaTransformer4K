param(
  [Parameter(Position = 0)]
  [string]$Task = "help",

  [int]$MaxSets = 200,
  [int]$Limit = 200,
  [int]$MaxWorkers = 3,
  [int]$Rpm = 60,

  [string]$Status = "ranked",
  # Default to high-quality pools; include loved because many piano/live-style maps land there.
  [string[]]$Statuses = @("ranked", "qualified", "loved"),
  [string]$Mode = "mania",
  [int]$Keys = 4,
  # "Live-ish" keyword queries (best-effort). You can override with -Queries.
  [string[]]$Queries = @("piano", "orchestra", "concert", "live", "acoustic", "Rousseau"),

  [string]$IdsFile = "data/osu2mir_audio/beatmapset_ids.txt",
  [string]$OszCache = "data/datasets2/osz_cache",
  [string]$Datasets2Dir = "data/datasets2",
  [string]$SampleDir = "data/datasets2_live_sample",

  [string]$Audio = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  return (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
}

function Require-VenvPython([string]$RepoRoot) {
  $py = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
  if (-not (Test-Path $py)) {
    throw "Missing venv. Create it first: python -m venv .venv; .\\.venv\\Scripts\\python -m pip install -U pip; .\\.venv\\Scripts\\python -m pip install requests"
  }
  return $py
}

function Show-Help {
  Write-Host "Tasks (run from repo root):"
  Write-Host "  .\\tasks.ps1 dataset:fetch-ids    [-Status ranked] [-Mode mania] [-Keys 4] [-MaxSets 200] [-Rpm 60]"
  Write-Host "  .\\tasks.ps1 dataset:bootstrap-live-sample [-SampleDir data/datasets2_live_sample] [-MaxSets 40] [-Statuses ranked qualified] [-Queries Rousseau,Animenz] [-Keys 4] [-Rpm 60]"
  Write-Host "  .\\tasks.ps1 dataset:download-osz [-IdsFile data/osu2mir_audio/beatmapset_ids.txt] [-OszCache data/datasets2/osz_cache] [-Limit 200] [-MaxWorkers 3]"
  Write-Host "  .\\tasks.ps1 dataset:build-d2     [-OszCache data/datasets2/osz_cache] [-Datasets2Dir data/datasets2]"
  Write-Host "  .\\tasks.ps1 dataset:classify     [-Datasets2Dir data/datasets2] [-Rpm 60]"
  Write-Host "  .\\tasks.ps1 ear:infer            -Audio <path-to-audio>"
  Write-Host ""
  Write-Host "Notes:"
  Write-Host "  - Put osu! API creds in config/osu_api.env (preferred) or data/osu2mir_audio/.env (fallback)."
  Write-Host "  - Mirror downloads go to data/datasets2/osz_cache (gitignored)."
}

$repoRoot = Resolve-RepoRoot

switch ($Task.ToLowerInvariant()) {
  "help" {
    Show-Help
    exit 0
  }
  "dataset:fetch-ids" {
    $py = Require-VenvPython $repoRoot
    & $py "data/osu2mir_audio/fetch_beatmapset_ids.py" --status $Status --mode $Mode --keys $Keys --max-sets $MaxSets --rpm $Rpm
    exit $LASTEXITCODE
  }
  "dataset:bootstrap-live-sample" {
    $py = Require-VenvPython $repoRoot

    $outIds = Join-Path $repoRoot "data\\osu2mir_audio\\beatmapset_ids.live_sample.txt"
    $tmpDir = Join-Path $repoRoot "reports\\tmp_live_sample"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

    # Fetch IDs for each query term (dedupe later).
    $perQuery = [Math]::Max(10, [Math]::Ceiling([double]$MaxSets / [double]([Math]::Max(1, $Queries.Count))) * 3)
    $idFiles = @()
    foreach ($q in $Queries) {
      $safe = ($q -replace '[^a-zA-Z0-9]+','_').Trim('_')
      $out = Join-Path $tmpDir ("ids_{0}.txt" -f $safe)
      $idFiles += $out
      & $py "data/osu2mir_audio/fetch_beatmapset_ids.py" --query $q --status $Statuses --mode $Mode --keys $Keys --max-sets $perQuery --rpm $Rpm --out $out
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    # Merge + de-dupe + take first MaxSets.
    $seen = New-Object 'System.Collections.Generic.HashSet[int]'
    $merged = New-Object System.Collections.Generic.List[int]
    foreach ($f in $idFiles) {
      if (-not (Test-Path $f)) { continue }
      foreach ($line in Get-Content -Path $f -ErrorAction SilentlyContinue) {
        $t = $line.Trim()
        if (-not $t) { continue }
        $id = 0
        if ([int]::TryParse($t, [ref]$id)) {
          if ($seen.Add($id)) { $merged.Add($id) | Out-Null }
        }
      }
    }
    if ($merged.Count -gt $MaxSets) {
      $merged = $merged.GetRange(0, $MaxSets)
    }
    if ($merged.Count -lt 1) {
      throw "No beatmapset IDs found for the provided Queries/Statuses."
    }

    Set-Content -Path $outIds -Value ($merged | ForEach-Object { $_.ToString() })
    Write-Host ("[OK] Wrote {0} unique beatmapset IDs: {1}" -f $merged.Count, $outIds)

    $sampleDirAbs = Join-Path $repoRoot $SampleDir
    $oszCache = Join-Path $sampleDirAbs "osz_cache"
    & $py "data/datasets2/download_osz_by_ids.py" --ids-file $outIds --out-dir $oszCache --limit $merged.Count --max-workers $MaxWorkers --rpm $Rpm
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $py "data/datasets2/osz_to_dataset.py" --input-dir $oszCache --output-dir $sampleDirAbs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $py "data/datasets2/clean_and_classify.py" --dataset-dir $sampleDirAbs --rpm $Rpm
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "[NEXT] Build Ear training data (deduped) from this sample:"
    Write-Host ("  python training\\scripts\\build_beat_this_osu_dataset.py --datasets2 `"{0}`" --dataset-name live_sample --allow-groups ranked,qualified --dedupe-by beatmapset" -f $SampleDir)
    exit 0
  }
  "dataset:download-osz" {
    $py = Require-VenvPython $repoRoot
    & $py "data/datasets2/download_osz_by_ids.py" --ids-file $IdsFile --out-dir $OszCache --limit $Limit --max-workers $MaxWorkers
    exit $LASTEXITCODE
  }
  "dataset:build-d2" {
    $py = Require-VenvPython $repoRoot
    & $py "data/datasets2/osz_to_dataset.py" --input-dir $OszCache --output-dir $Datasets2Dir
    exit $LASTEXITCODE
  }
  "dataset:classify" {
    $py = Require-VenvPython $repoRoot
    & $py "data/datasets2/clean_and_classify.py" --dataset-dir $Datasets2Dir --rpm $Rpm
    exit $LASTEXITCODE
  }
  "ear:infer" {
    if (-not $Audio) { throw "Missing -Audio" }
    $py = Require-VenvPython $repoRoot
    & $py "modules/beat_this/beat_to_osu.py" $Audio --timing-profile auto -o out.timing.txt
    exit $LASTEXITCODE
  }
  default {
    Write-Host "[ERROR] Unknown task: $Task"
    Show-Help
    exit 1
  }
}
