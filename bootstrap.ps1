<#
.SYNOPSIS
    From-scratch installer: install Git, clone (or update) the repo, then run setup.ps1.

.DESCRIPTION
    The single entry point for a brand-new PC. It ensures Git (via winget), clones the
    repository, and hands off to setup.ps1 (which installs Python 3.11 + all deps +
    ffmpeg + the boot auto-start shortcut). After this, only the per-device steps remain
    (model file, PIN/Supabase/cameras, BIOS + Windows auto-login) — see INSTALL.md.

    Note: cloning a PRIVATE repo will pop a one-time GitHub login (browser). winget may
    pop a UAC prompt while installing Git/Python — both are expected.

.PARAMETER Dir      Target folder (default C:\HGCameraCounter).
.PARAMETER Branch   Branch to install (default main).
.PARAMETER Repo     Git URL.
.PARAMETER Watchdog Also install the liveness watchdog (passed to setup.ps1).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File bootstrap.ps1
    powershell -ExecutionPolicy Bypass -File bootstrap.ps1 -Dir D:\HGCC -Watchdog
#>
[CmdletBinding()]
param(
    [string]$Dir = "C:\HGCameraCounter",
    [string]$Branch = "main",
    [string]$Repo = "https://github.com/huagruay-git/HGCameraCounter.git",
    [switch]$Watchdog
)

$ErrorActionPreference = 'Continue'
function Refresh-Path {
    $m = [Environment]::GetEnvironmentVariable('Path','Machine')
    $u = [Environment]::GetEnvironmentVariable('Path','User')
    $env:Path = (@($m, $u) | Where-Object { $_ }) -join ';'
}
function Have($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

Write-Host "HG Camera Counter - bootstrap installer" -ForegroundColor White

# 1) Git (needed to clone). Install via winget if missing.
if (-not (Have git)) {
    Write-Host "Installing Git..." -ForegroundColor Cyan
    if (-not (Have winget)) { Write-Host "ERROR: winget unavailable - install Git manually then re-run." -ForegroundColor Red; exit 1 }
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements --disable-interactivity
    Refresh-Path
}
if (-not (Have git)) { Write-Host "ERROR: Git still not found - reopen the terminal and re-run." -ForegroundColor Red; exit 1 }

# 2) Clone, or update if it already exists.
if (Test-Path (Join-Path $Dir ".git")) {
    Write-Host "Updating existing repo at $Dir" -ForegroundColor Cyan
    git -C $Dir fetch origin
    git -C $Dir checkout $Branch
    git -C $Dir pull --ff-only
} else {
    Write-Host "Cloning into $Dir (a GitHub login may appear)..." -ForegroundColor Cyan
    git clone --branch $Branch $Repo $Dir
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: clone failed (exit $LASTEXITCODE)." -ForegroundColor Red; exit 1 }
}

# 3) Hand off to the main installer.
Set-Location $Dir
$setupArgs = @()
if ($Watchdog) { $setupArgs += '-Watchdog' }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Dir "setup.ps1") @setupArgs
