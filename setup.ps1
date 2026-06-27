<#
.SYNOPSIS
    One-command install of HG Camera Counter from source on a Windows PC.

.DESCRIPTION
    Run this once in a freshly cloned repo and it provisions everything needed to run
    the counter from source:
      1. locate Python 3.11 (the supported interpreter)
      2. create the .venv virtual environment
      3. install all Python dependencies (requirements.txt)
      4. make sure ffmpeg is available (downloads it if missing)
      5. create data/config/config.yaml from the template if absent
      6. install the boot auto-start shortcut (scripts/install_autostart.ps1)
      7. optionally install the liveness watchdog scheduled task (-Watchdog, needs admin)

    It is idempotent: an existing venv/config is reused, not clobbered. Two things it
    CANNOT do (they are per-device and never in git) and will remind you about:
      - the trained model models\best.pt
      - real Supabase secrets + the login PIN (provision + first manual launch)

.PARAMETER Python
    Path to a python.exe to build the venv with. Default: auto-detect via `py -3.11`.
.PARAMETER Recreate
    Delete and rebuild the .venv from scratch.
.PARAMETER SkipDeps
    Skip the pip install step (use when dependencies are already installed).
.PARAMETER Watchdog
    Also install the auto-reboot liveness watchdog task (run elevated for this).
.PARAMETER NoAutostart
    Do not install the boot auto-start shortcut.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Watchdog
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Recreate
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$Recreate,
    [switch]$SkipDeps,
    [switch]$Watchdog,
    [switch]$NoAutostart
)

# Native tools (py/pip) write progress to stderr; under 'Stop' that can abort the run.
# Use 'Continue' and check $LASTEXITCODE explicitly after each critical native call.
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    [ok] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    [!] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

Write-Host "HG Camera Counter - source install" -ForegroundColor White
Write-Host "Repo: $root"

# 1) Decide how to invoke the base Python 3.11 (avoid capturing its path: a Thai user
#    path comes back mis-encoded from a native exe and breaks Test-Path).
Step "Locating Python 3.11"
if ($Python) {
    if (-not (Test-Path $Python)) { Die "Given -Python path not found: $Python" }
    $pyBin = $Python; $pyPre = @()
} else {
    $pyBin = 'py'; $pyPre = @('-3.11')
}
$ver = $null
try { $ver = (& $pyBin @pyPre -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null) } catch { $ver = $null }
if (-not $ver) { Die "Python 3.11 not found. Install 3.11.x from python.org or pass -Python <path to python.exe>." }
$ver = ("$ver").Trim()
if ($ver -ne "3.11") { Warn "Interpreter reports Python $ver (3.11 is the tested version) - continuing." }
Ok "Python $ver available"

# 2) Create the virtual environment.
Step "Creating virtual environment (.venv)"
$venv = Join-Path $root ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"
if ($Recreate -and (Test-Path $venv)) {
    Warn "Removing existing .venv (-Recreate)"
    Remove-Item $venv -Recurse -Force -ErrorAction Stop
}
if (-not (Test-Path $venvPy)) {
    & $pyBin @pyPre -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) { Die "venv creation failed (exit $LASTEXITCODE)." }
    Ok "Created .venv"
} else {
    Ok ".venv already exists (reused)"
}

# 3) Install Python dependencies.
if ($SkipDeps) {
    Warn "Skipping dependency install (-SkipDeps)"
} else {
    Step "Installing dependencies (this can take several minutes - torch is large)"
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r (Join-Path $root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Die "pip install failed (exit $LASTEXITCODE). See output above." }
    Ok "Dependencies installed"
}

# 4) Ensure ffmpeg (used by the recorder). Downloads a build if none is found.
Step "Checking ffmpeg"
Push-Location $root
$code = "from shared.ffmpeg_manager import ensure_ffmpeg_available as e; p,s=e(auto_download=True); print('FFMPEG_RESULT', s, p)"
& $venvPy -c $code
$ffOk = ($LASTEXITCODE -eq 0)
Pop-Location
if ($ffOk) { Ok "ffmpeg ready" } else { Warn "Could not auto-provision ffmpeg. Install it or set runtime.ffmpeg_path later." }

# 5) Config file: create from template only if missing (never overwrite a real one).
Step "Checking config"
$cfg = Join-Path $root "data\config\config.yaml"
$tpl = Join-Path $root "data\config\config.template.yaml"
if (Test-Path $cfg) {
    Ok "data\config\config.yaml exists (left untouched)"
} elseif (Test-Path $tpl) {
    New-Item -ItemType Directory -Force (Split-Path $cfg) | Out-Null
    Copy-Item $tpl $cfg -ErrorAction Stop
    Warn "Created config.yaml from template - fill in Supabase URL/key, branch_code, cameras."
    Warn "Then encrypt secrets: .venv\Scripts\python.exe scripts\encrypt_config_secrets.py"
} else {
    Warn "No config or template under data\config - provision config.yaml manually."
}

# 6) Boot auto-start shortcut.
if ($NoAutostart) {
    Warn "Skipping auto-start shortcut (-NoAutostart)"
} else {
    Step "Installing boot auto-start shortcut"
    try { & (Join-Path $root "scripts\install_autostart.ps1") }
    catch { Warn "Auto-start install failed: $($_.Exception.Message)" }
}

# 7) Optional liveness watchdog scheduled task.
if ($Watchdog) {
    Step "Installing liveness watchdog task (needs admin)"
    try { & (Join-Path $root "scripts\install_watchdog_task.ps1"); Ok "Watchdog task installed" }
    catch { Warn "Watchdog install failed ($($_.Exception.Message)). Re-run elevated: scripts\install_watchdog_task.ps1" }
}

# Summary + remaining manual steps.
Step "Done - remaining manual steps"
if (-not (Test-Path (Join-Path $root "models\best.pt"))) {
    Warn "models\best.pt is missing - copy the trained model in (not in git)."
}
Write-Host ""
Write-Host "Next:" -ForegroundColor White
Write-Host "  1. Provision data\config\config.yaml (Supabase, branch, cameras) + encrypt secrets"
Write-Host "  2. Put the trained model at models\best.pt"
Write-Host "  3. Launch once to set the PIN + bind this machine:"
Write-Host "       .venv\Scripts\python.exe controller\main.py"
Write-Host "  4. (device) BIOS 'Restore on AC Power Loss' = Power On + Windows auto-login"
Write-Host ""
Write-Host "After that, the app auto-starts and resumes counting on every boot." -ForegroundColor Green
