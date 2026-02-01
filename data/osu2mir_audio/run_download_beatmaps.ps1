param(
  [int]$Limit = 50,
  [int]$MaxWorkers = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$downloader = Join-Path $PSScriptRoot "download_beatmaps.py"

if (-not (Test-Path $venvPython)) {
  throw "Missing venv python at '$venvPython'. Create it first: python -m venv .venv"
}

if (-not (Test-Path $downloader)) {
  throw "Missing downloader at '$downloader'"
}

Write-Host "osu! API v2 credentials"
$envFile = $null
$repoEnvFile = Join-Path $repoRoot "config\\osu_api.env"
$localEnvFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $repoEnvFile) { $envFile = $repoEnvFile }
elseif (Test-Path $localEnvFile) { $envFile = $localEnvFile }

$clientId = $null
$clientSecretPlain = $null

if ($envFile -and (Test-Path $envFile)) {
  Write-Host "  - Loading from $envFile"
  foreach ($line in Get-Content -Path $envFile) {
    $t = $line.Trim()
    if (-not $t -or $t.StartsWith('#')) { continue }
    $eq = $t.IndexOf('=')
    if ($eq -lt 1) { continue }
    $k = $t.Substring(0, $eq).Trim()
    $v = $t.Substring($eq + 1).Trim()
    if ($v.Length -ge 2 -and (($v[0] -eq '"' -and $v[$v.Length-1] -eq '"') -or ($v[0] -eq "'" -and $v[$v.Length-1] -eq "'"))) {
      $v = $v.Substring(1, $v.Length - 2)
    }
    if ($k -eq "OSU_CLIENT_ID") { $clientId = $v }
    if ($k -eq "OSU_CLIENT_SECRET") { $clientSecretPlain = $v }
  }
}

if (-not $clientId -or -not $clientSecretPlain) {
  Write-Host "  - Client ID: visible input"
  Write-Host "  - Client Secret: hidden input"
  $clientId = Read-Host "Client ID"
  $clientSecretSecure = Read-Host "Client Secret" -AsSecureString
  $clientSecretPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($clientSecretSecure))
}

try {
  $env:OSU_CLIENT_ID = $clientId
  $env:OSU_CLIENT_SECRET = $clientSecretPlain

  & $venvPython $downloader --limit $Limit --max-workers $MaxWorkers
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Remove-Item Env:OSU_CLIENT_ID -ErrorAction SilentlyContinue
  Remove-Item Env:OSU_CLIENT_SECRET -ErrorAction SilentlyContinue
}
