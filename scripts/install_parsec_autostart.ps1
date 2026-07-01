<#
.SYNOPSIS
    Add Parsec to the Windows Startup folder so it launches at logon (for remote access
    after a reboot), independent of Parsec's own "run on startup" setting.

.DESCRIPTION
    Some Parsec builds hide the "Run on startup" toggle. This drops a Startup-folder
    shortcut to parsecd.exe so a rebooted branch PC always brings Parsec back up (paired
    with Windows auto-login + the app's own boot auto-start). Idempotent; -Uninstall removes it.

.PARAMETER Path      Explicit path to parsecd.exe (default: auto-detect common locations).
.PARAMETER Uninstall Remove the Parsec startup shortcut.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_parsec_autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_parsec_autostart.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string]$Path = "",
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
$startup = [Environment]::GetFolderPath('Startup')
$link = Join-Path $startup 'Parsec.lnk'

if ($Uninstall) {
    if (Test-Path $link) {
        Remove-Item $link -Force
        Write-Host "Removed Parsec startup shortcut: $link" -ForegroundColor Green
    } else {
        Write-Host "No Parsec startup shortcut found." -ForegroundColor Yellow
    }
    return
}

$candidates = @(
    $Path,
    "C:\Program Files\Parsec\parsecd.exe",
    (Join-Path $env:ProgramFiles 'Parsec\parsecd.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Parsec\parsecd.exe'),
    (Join-Path $env:LOCALAPPDATA 'Parsec\parsecd.exe'),
    (Join-Path $env:APPDATA 'Parsec\parsecd.exe')
) | Where-Object { $_ }

$exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) {
    throw "parsecd.exe not found. Install Parsec, or pass -Path 'C:\Program Files\Parsec\parsecd.exe'."
}

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath       = $exe
$sc.WorkingDirectory = (Split-Path $exe)
$sc.Description       = 'Parsec - auto-start at logon for remote access'
$sc.WindowStyle       = 7   # minimized
$sc.Save()

Write-Host "Installed Parsec startup shortcut:" -ForegroundColor Green
Write-Host "  Shortcut : $link"
Write-Host "  Target   : $exe"
Write-Host "It will launch Parsec at the next logon."
Write-Host "To remove: powershell -ExecutionPolicy Bypass -File scripts\install_parsec_autostart.ps1 -Uninstall"
