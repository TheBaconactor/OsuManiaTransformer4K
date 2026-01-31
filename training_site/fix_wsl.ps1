param(
  [switch]$ReinstallWSL = $true,
  [switch]$UninstallOnly,
  [switch]$InstallOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-HasProperty([object]$Obj, [string]$Name) {
  if ($null -eq $Obj) { return $false }
  return ($null -ne $Obj.PSObject.Properties[$Name])
}

function Test-IsAdmin {
  $current = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($current)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-WSLProducts {
  $paths = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
    'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
  )
  foreach ($p in $paths) {
    Get-ItemProperty $p -ErrorAction SilentlyContinue |
      Where-Object { (Test-HasProperty $_ 'DisplayName') -and ($_.DisplayName -like '*Windows Subsystem for Linux*' -or $_.DisplayName -like '*WSLg*') } |
      Select-Object DisplayName, DisplayVersion, PSChildName
  }
}

if (-not (Test-IsAdmin)) {
  Write-Host "[INFO] Elevation required. Relaunching as Administrator..."
  try {
    Start-Process -FilePath "powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`" -ReinstallWSL:$ReinstallWSL" -Verb RunAs -ErrorAction Stop
    exit 0
  } catch {
    Write-Host "[ERROR] Elevation cancelled or failed. Re-run this script as Administrator."
    exit 1
  }
}

if ($UninstallOnly -and $InstallOnly) {
  Write-Host "[ERROR] Use only one of -UninstallOnly or -InstallOnly."
  exit 1
}

if ($InstallOnly) {
  Write-Host "[INFO] Install-only mode."
}

if ($UninstallOnly) {
  Write-Host "[INFO] Uninstall-only mode."
}

Write-Host "[1/5] Stop any stuck Windows Installer processes"
Get-Process msiexec -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

if (-not $InstallOnly) {
  Write-Host "[2/5] Stop WSL and services"
  try { wsl --shutdown | Out-Null } catch {}
  try { Stop-Service -Name LxssManager -Force -ErrorAction SilentlyContinue } catch {}

  Write-Host "[3/5] Remove old WSL MSI products (if present)"
  $guids = @(
    "{D726B73B-1F51-4144-943E-09BFA1C409D0}", # Windows Subsystem for Linux
    "{F8474A47-8B5D-4466-ACE3-78EAB3BF21A8}", # Windows Subsystem for Linux Update
    "{3CBDE512-7510-4F90-B1C0-7C4EB9DD7C26}"  # WSLg Preview
  )
  foreach ($g in $guids) {
    $p = Start-Process -FilePath "msiexec.exe" -ArgumentList "/x $g /qn /norestart" -Wait -NoNewWindow -PassThru -ErrorAction SilentlyContinue
    if ($p -and $p.ExitCode -ne 0) {
      Write-Host "[WARN] msiexec /x $g returned exit code $($p.ExitCode)"
      if ($p.ExitCode -eq 1612) {
        Write-Host "[WARN] Exit code 1612 means Windows Installer can't find the cached MSI for that product."
        Write-Host "[WARN] This usually requires manual cleanup (Windows troubleshooter) before WSL can be reinstalled/updated."
      }
    }
  }

  Write-Host "[4/5] Remove WSL AppX (if present)"
  Get-AppxPackage MicrosoftCorporationII.WindowsSubsystemForLinux -AllUsers -ErrorAction SilentlyContinue | Remove-AppxPackage -AllUsers -ErrorAction SilentlyContinue
  Get-AppxPackage MicrosoftCorporationII.WindowsSubsystemForLinux -ErrorAction SilentlyContinue | Remove-AppxPackage -ErrorAction SilentlyContinue

  $remaining = @(Get-WSLProducts)
  if ($remaining.Count -gt 0) {
    Write-Host "[WARN] Some WSL MSI entries still present:"
    $remaining | ForEach-Object { Write-Host ("  - {0} {1} {2}" -f $_.DisplayName, $_.DisplayVersion, $_.PSChildName) }
    Write-Host "[WARN] Reboot Windows, then re-run this script with -InstallOnly."
    exit 2
  }

  if ($UninstallOnly) {
    Write-Host "[OK] WSL components removed. Reboot Windows, then run:"
    Write-Host '  & "C:\Users\troll\Desktop\OsuManiaTransformer4K\training_site\fix_wsl.ps1" -InstallOnly'
    exit 0
  }
}

if ($ReinstallWSL) {
  Write-Host "[5/5] Install WSL via winget"
  winget install -e --id Microsoft.WSL --source winget --accept-package-agreements --accept-source-agreements
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] winget failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }

  Write-Host "[6/6] Update WSL and shutdown"
  wsl --update --web-download
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] wsl --update failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
  wsl --shutdown
  Write-Host "[OK] WSL reinstalled. Reboot Windows, then re-run diagnostics:"
  Write-Host "  wsl --version"
  Write-Host '  wsl bash -lc "./training_site/diagnose_rocm_wsl.sh"'
} else {
  Write-Host "[OK] WSL components removed. Reinstall manually when ready."
}
