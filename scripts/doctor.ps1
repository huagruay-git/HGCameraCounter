<#
.SYNOPSIS
    System Doctor - check everything HG Camera Counter needs and (optionally) auto-install
    what's missing. Works for a fresh install and for already-installed machines.

.DESCRIPTION
    Checks: Visual C++ Redistributable, Git, the .venv + Python packages (torch/opencv/
    ultralytics/PySide6/supabase), ffmpeg, the AI model, config, and the boot/Parsec
    shortcuts. With -Fix it installs/repairs the fixable ones (VC++ + Git via winget,
    Python deps via pip, ffmpeg download, model download via the update manifest, and the
    Startup shortcuts). Reports the rest (config needs provisioning per device).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1          # check only
    powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1 -Fix     # check + auto-fix
#>
[CmdletBinding()]
param([switch]$Fix)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

$script:issues = 0
$script:fixed = 0
function Ok($m)   { Write-Host "  [OK]   $m" -ForegroundColor Green }
function Bad($m)  { Write-Host "  [MISS] $m" -ForegroundColor Yellow; $script:issues++ }
function Did($m)  { Write-Host "  [FIX]  $m" -ForegroundColor Cyan; $script:fixed++ }
function Step($m) { Write-Host "`n== $m ==" -ForegroundColor White }
function Have($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }
function Refresh-Path {
    $env:Path = (@([Environment]::GetEnvironmentVariable('Path','Machine'),
                   [Environment]::GetEnvironmentVariable('Path','User')) | Where-Object { $_ }) -join ';'
}

Write-Host "HG Camera Counter - System Doctor" -ForegroundColor White
Write-Host "Repo: $root   (mode: $(if($Fix){'CHECK + FIX'}else{'CHECK ONLY'}))"

# 1) Visual C++ Redistributable (torch/opencv DLLs need it)
Step "Visual C++ Redistributable"
$vc = $false
try {
    $vc = ((Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction Stop).Installed -eq 1)
} catch { $vc = $false }
if ($vc) { Ok "installed" }
else {
    Bad "not installed (torch/opencv will fail with a DLL error)"
    if ($Fix -and (Have winget)) {
        winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements --disable-interactivity
        Did "Visual C++ Redistributable installed"
    } elseif ($Fix) { Write-Host "     download: https://aka.ms/vs/16/release/vc_redist.x64.exe" }
}

# 2) Git (for updates)
Step "Git"
if (Have git) { Ok "installed" }
else {
    Bad "not installed (needed for updates)"
    if ($Fix -and (Have winget)) {
        winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements --disable-interactivity
        Refresh-Path; Did "Git installed"
    }
}

# 3) Python venv
Step "Python virtual environment"
if (Test-Path $venvPy) { Ok ".venv present" }
else {
    Bad ".venv missing - run setup.bat to create it"
    Write-Host "Cannot check Python packages without the venv. Stopping fixable-deps checks." -ForegroundColor Yellow
}

# 4) Python packages
if (Test-Path $venvPy) {
    Step "Python packages (torch / opencv / ultralytics / PySide6 / supabase)"
    $probe = "import importlib.util; mods=['torch','cv2','numpy','ultralytics','PySide6','supabase','yaml']; bad=[m for m in mods if importlib.util.find_spec(m) is None]; print('MISSING:'+','.join(bad) if bad else 'ALLOK')"
    $res = (& $venvPy -c $probe 2>&1) -join "`n"
    if ($res -match 'ALLOK') { Ok "all key packages import" }
    else {
        Bad "some packages missing/broken: $res"
        if ($Fix) {
            Push-Location $root
            & $venvPy -m pip install -r (Join-Path $root "requirements.txt")
            Pop-Location
            Did "reinstalled requirements.txt"
        }
    }

    # 5) ffmpeg
    Step "ffmpeg"
    Push-Location $root
    $ff = (& $venvPy -c "from shared.ffmpeg_manager import ensure_ffmpeg_available as e; p,s=e(auto_download=$(if($Fix){'True'}else{'False'})); print('FFOK' if p else 'FFMISS')" 2>&1) -join "`n"
    Pop-Location
    if ($ff -match 'FFOK') { if ($Fix) { Did "ffmpeg ready" } else { Ok "ffmpeg ready" } }
    else { Bad "ffmpeg not found (run with -Fix to download)" }

    # 6) AI model
    Step "AI model (models\best.pt)"
    $model = Join-Path $root "models\best.pt"
    if (Test-Path $model) { Ok "present" }
    else {
        Bad "models\best.pt missing"
        if ($Fix) {
            Push-Location $root
            $dl = "import sys; from runtime.model_ota import fetch_model_manifest, apply_model_from_entry; from shared.config import DEFAULT_MODELS_MANIFEST_URL as U; " +
                  "m=fetch_model_manifest(U).get('models',[]); " +
                  "pick=next((x for x in m if x.get('recommended')), (m[0] if m else None)); " +
                  "print('NOMODEL') if not pick else (apply_model_from_entry(pick,'models/best.pt'), print('MODELOK'))"
            $mres = (& $venvPy -c $dl 2>&1) -join "`n"
            Pop-Location
            if ($mres -match 'MODELOK') { Did "model downloaded from the update server" }
            else { Write-Host "     could not auto-download model ($mres) - use the อัปเดตโมเดล tab or copy best.pt" -ForegroundColor Yellow }
        }
    }
}

# 7) Config (provisioned per device - report only)
Step "Config"
if (Test-Path (Join-Path $root "data\config\config.yaml")) { Ok "config.yaml present" }
else { Bad "config.yaml missing - open the app and use the Setup Wizard / Cloud Sync" }

# 8) Startup shortcuts (auto-start + Parsec)
Step "Startup shortcuts"
$startup = [Environment]::GetFolderPath('Startup')
if (Test-Path (Join-Path $startup 'HG Camera Counter.lnk')) { Ok "app auto-start shortcut present" }
else {
    Bad "app auto-start shortcut missing"
    if ($Fix) { & (Join-Path $root "scripts\install_autostart.ps1") | Out-Null; Did "auto-start shortcut created" }
}
if (Test-Path (Join-Path $startup 'Parsec.lnk')) { Ok "Parsec auto-start shortcut present" }
elseif (Test-Path "C:\Program Files\Parsec\parsecd.exe") {
    Bad "Parsec installed but not in Startup"
    if ($Fix) { & (Join-Path $root "scripts\install_parsec_autostart.ps1") | Out-Null; Did "Parsec auto-start shortcut created" }
} else { Write-Host "  [--]   Parsec not installed (skip)" -ForegroundColor DarkGray }

# Summary
Write-Host "`n=================================" -ForegroundColor White
if ($script:issues -eq 0) { Write-Host "All good - nothing missing." -ForegroundColor Green }
elseif ($Fix) { Write-Host "Found $script:issues issue(s), fixed $script:fixed. Re-run to verify; restart the app." -ForegroundColor Cyan }
else { Write-Host "Found $script:issues issue(s). Run again with -Fix to auto-install." -ForegroundColor Yellow }
