param(
  [int]$Limit = 50,
  [int]$MaxWorkers = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$downloader = Join-Path $PSScriptRoot "download_beatmaps.py"

if (-not (Test-Path $venvPython)) {
  throw "Missing venv python at '$venvPython'. Create it first: python -m venv .venv"
}

if (-not (Test-Path $downloader)) {
  throw "Missing downloader at '$downloader'"
}

Write-Host "osu! API v2 credentials"
Write-Host "  - Client ID: visible input"
Write-Host "  - Client Secret: hidden input"

$clientId = Read-Host "Client ID"
$clientSecretSecure = Read-Host "Client Secret" -AsSecureString
$clientSecretPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($clientSecretSecure))

try {
  $env:OSU_CLIENT_ID = $clientId
  $env:OSU_CLIENT_SECRET = $clientSecretPlain

  & $venvPython $downloader --limit $Limit --max-workers $MaxWorkers
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Remove-Item Env:OSU_CLIENT_ID -ErrorAction SilentlyContinue
  Remove-Item Env:OSU_CLIENT_SECRET -ErrorAction SilentlyContinue
}

