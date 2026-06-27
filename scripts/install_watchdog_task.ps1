<#
.SYNOPSIS
  Register (or remove) the HGCC liveness watchdog as a Windows Scheduled Task that
  starts at boot, runs as SYSTEM, and auto-restarts if the watchdog process dies.

  The watchdog (runtime/watchdog.py) watches runtime/dashboard_state.json and reboots
  the PC if the counting runtime freezes/dies. It only ACTS when watchdog.enabled: true
  is set in data/config/config.yaml — installing the task alone changes nothing until
  you flip that flag.

.PARAMETER Target
  Path to HGCameraCounter.exe (frozen build) OR a python.exe to run the source module.
  Omit to auto-detect: built exe -> installed exe -> source venv python.

.PARAMETER TaskName
  Scheduled Task name. Default: "HGCC Watchdog".

.PARAMETER Uninstall
  Remove the task instead of installing it.

.EXAMPLE
  # Auto-detect and install (run from an ELEVATED PowerShell):
  powershell -ExecutionPolicy Bypass -File scripts\install_watchdog_task.ps1

.EXAMPLE
  # Point at a specific exe:
  .\scripts\install_watchdog_task.ps1 -Target "C:\Users\me\AppData\Local\HGCameraCounter\HGCameraCounter.exe"

.EXAMPLE
  # Remove it:
  .\scripts\install_watchdog_task.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string]$Target,
    [string]$TaskName = "HGCC Watchdog",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Error "This script must run in an ELEVATED PowerShell (Run as Administrator)."
    exit 1
}

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    } else {
        Write-Host "No scheduled task named '$TaskName' found." -ForegroundColor Yellow
    }
    exit 0
}

# --- repo root = parent of this scripts/ folder -----------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot

# --- resolve what to run ----------------------------------------------------
$Program = $null
$Arguments = $null
$WorkDir = $null

function Resolve-Program([string]$t) {
    if ([string]::IsNullOrWhiteSpace($t)) { return $null }
    if (Test-Path $t) { return (Resolve-Path $t).Path }
    return $null
}

if ($Target) {
    $resolved = Resolve-Program $Target
    if (-not $resolved) { Write-Error "Target not found: $Target"; exit 1 }
    if ($resolved -match '(?i)python(w)?\.exe$') {
        $Program = $resolved
        $Arguments = "`"$RepoRoot\runtime\watchdog.py`""
        $WorkDir = $RepoRoot
    } else {
        $Program = $resolved
        $Arguments = "--watchdog"
        $WorkDir = Split-Path -Parent $resolved
    }
} else {
    # Auto-detect, in priority order.
    $candidates = @(
        (Join-Path $RepoRoot "dist\HGCameraCounter\HGCameraCounter.exe"),
        (Join-Path $env:LOCALAPPDATA "HGCameraCounter\HGCameraCounter.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $Program = (Resolve-Path $c).Path; $Arguments = "--watchdog"; $WorkDir = Split-Path -Parent $Program; break }
    }
    if (-not $Program) {
        # Fall back to source via the project venv.
        $py = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
        if (-not (Test-Path $py)) { $py = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
        if (-not (Test-Path $py)) { Write-Error "No HGCameraCounter.exe and no .venv python found. Pass -Target explicitly."; exit 1 }
        $Program = (Resolve-Path $py).Path
        $Arguments = "`"$RepoRoot\runtime\watchdog.py`""
        $WorkDir = $RepoRoot
    }
}

Write-Host "Program  : $Program"
Write-Host "Arguments: $Arguments"
Write-Host "WorkDir  : $WorkDir"

# --- build + register the task ----------------------------------------------
$action  = New-ScheduledTaskAction -Execute $Program -Argument $Arguments -WorkingDirectory $WorkDir
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName' (runs at boot as SYSTEM, auto-restarts)." -ForegroundColor Green

# Start it now so you don't have to reboot to begin monitoring.
Start-ScheduledTask -TaskName $TaskName
Write-Host "Started '$TaskName'." -ForegroundColor Green

Write-Host ""
Write-Host "NEXT: enable it in data\config\config.yaml ->  watchdog:\n        enabled: true" -ForegroundColor Cyan
Write-Host "Verify a healthy machine first:  python runtime\watchdog.py --check" -ForegroundColor Cyan
Write-Host "Logs: logs\watchdog.log" -ForegroundColor Cyan
