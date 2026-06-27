<#
.SYNOPSIS
    Install (or remove) an auto-start shortcut so HG Camera Counter launches itself
    when this user logs in, then resumes counting unattended.

.DESCRIPTION
    Creates a shortcut in the current user's Startup folder that runs the dashboard
    from source with --autostart. On a power cut, once the BIOS powers the PC back on
    and Windows auto-logs-in, the dashboard opens, the PIN is skipped (device stays
    machine-bound for anti-theft), and counting starts automatically.

    A Startup-folder shortcut is used instead of Task Scheduler on purpose: the Thai
    user path (...\พีซี\...) has been mangled in the Task Scheduler SYSTEM context
    before, but a Startup shortcut runs in the normal logged-in user session where
    the path resolves correctly.

.PARAMETER Uninstall
    Remove the shortcut instead of creating it.

.PARAMETER Console
    Use python.exe (shows a console window) instead of pythonw.exe. Handy for
    debugging the first unattended launch; omit for normal silent operation.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$Console
)

$ErrorActionPreference = 'Stop'

# Project root = parent of this scripts\ folder.
$root = Split-Path -Parent $PSScriptRoot
$startup = [Environment]::GetFolderPath('Startup')
$linkPath = Join-Path $startup 'HG Camera Counter.lnk'

if ($Uninstall) {
    if (Test-Path $linkPath) {
        Remove-Item $linkPath -Force
        Write-Host "Removed auto-start shortcut: $linkPath" -ForegroundColor Green
    } else {
        Write-Host "No auto-start shortcut found at: $linkPath" -ForegroundColor Yellow
    }
    return
}

# Pick the interpreter from the project venv (Python 3.11 with all deps).
$exeName = if ($Console) { 'python.exe' } else { 'pythonw.exe' }
$pyExe = Join-Path $root ".venv\Scripts\$exeName"
if (-not (Test-Path $pyExe)) {
    throw "Interpreter not found: $pyExe`nCreate the venv first, or pass -Console / check the path."
}

$entry = Join-Path $root 'controller\main.py'
if (-not (Test-Path $entry)) {
    throw "Entry point not found: $entry"
}

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($linkPath)
$sc.TargetPath       = $pyExe
$sc.Arguments        = 'controller\main.py --autostart'
$sc.WorkingDirectory = $root
$sc.WindowStyle      = 1
$sc.Description      = 'HG Camera Counter - auto-start counting at logon'
$sc.IconLocation     = "$pyExe,0"
$sc.Save()

Write-Host "Installed auto-start shortcut:" -ForegroundColor Green
Write-Host "  Shortcut : $linkPath"
Write-Host "  Runs     : `"$pyExe`" controller\main.py --autostart"
Write-Host "  StartIn  : $root"
Write-Host ""
Write-Host "It will launch at the next logon. To test now, run:" -ForegroundColor Cyan
Write-Host "  & `"$pyExe`" controller\main.py --autostart   (from $root)"
Write-Host "To remove: powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Uninstall"
