param(
  [Parameter(Position = 0)]
  [string]$Task = "help",

  [int]$MaxSets = 200,
  [int]$Limit = 200,
  [int]$MaxWorkers = 3,
  [int]$Rpm = 60,

  [string]$Status = "ranked",
  [string]$Mode = "mania",
  [int]$Keys = 4,

  [string]$IdsFile = "Datasets/beatmapset_ids.txt",
  [string]$OszCache = "Datasets2/osz_cache",
  [string]$Datasets2Dir = "Datasets2",

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
  Write-Host "  .\\tasks.ps1 dataset:download-osz [-IdsFile Datasets/beatmapset_ids.txt] [-OszCache Datasets2/osz_cache] [-Limit 200] [-MaxWorkers 3]"
  Write-Host "  .\\tasks.ps1 dataset:build-d2     [-OszCache Datasets2/osz_cache] [-Datasets2Dir Datasets2]"
  Write-Host "  .\\tasks.ps1 dataset:classify     [-Datasets2Dir Datasets2] [-Rpm 60]"
  Write-Host "  .\\tasks.ps1 ear:infer            -Audio <path-to-audio>"
  Write-Host ""
  Write-Host "Notes:"
  Write-Host "  - Put osu! API creds in Datasets/.env (see Datasets/.env.example)."
  Write-Host "  - Mirror downloads go to Datasets2/osz_cache (gitignored)."
}

$repoRoot = Resolve-RepoRoot

switch ($Task.ToLowerInvariant()) {
  "help" {
    Show-Help
    exit 0
  }
  "dataset:fetch-ids" {
    $py = Require-VenvPython $repoRoot
    & $py "Datasets/fetch_beatmapset_ids.py" --status $Status --mode $Mode --keys $Keys --max-sets $MaxSets --rpm $Rpm
    exit $LASTEXITCODE
  }
  "dataset:download-osz" {
    $py = Require-VenvPython $repoRoot
    & $py "Datasets2/download_osz_by_ids.py" --ids-file $IdsFile --out-dir $OszCache --limit $Limit --max-workers $MaxWorkers
    exit $LASTEXITCODE
  }
  "dataset:build-d2" {
    $py = Require-VenvPython $repoRoot
    & $py "Datasets2/osz_to_dataset.py" --input-dir $OszCache --output-dir $Datasets2Dir
    exit $LASTEXITCODE
  }
  "dataset:classify" {
    $py = Require-VenvPython $repoRoot
    & $py "Datasets2/clean_and_classify.py" --dataset-dir $Datasets2Dir --rpm $Rpm
    exit $LASTEXITCODE
  }
  "ear:infer" {
    if (-not $Audio) { throw "Missing -Audio" }
    python "Modules/beat_this/beat_to_osu.py" $Audio --timing-profile auto -o out.timing.txt
    exit $LASTEXITCODE
  }
  default {
    Write-Host "[ERROR] Unknown task: $Task"
    Show-Help
    exit 1
  }
}

