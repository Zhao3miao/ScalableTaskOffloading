# ============================================================
# Update subway_station_scenarios/*.sh paths and model names
# ------------------------------------------------------------
# Replaces:
#   1) Scenario path:
#      subway_station_scenarios/subway_station_max_ue_
#        -> subway_station_scenarios/subway_station_max_ue_
#   2) Model name suffix (avoid checkpoint collision with v1):
#      subway50 -> subway50
#
# Usage (from repo root or this folder):
#   pwsh ./subway_station_scenarios/update_paths.ps1
#   pwsh ./subway_station_scenarios/update_paths.ps1 -DryRun
# ============================================================

[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$shFiles = Get-ChildItem -Path $scriptDir -Filter '*.sh' -File

$replacements = @(
    @{
        Old = 'subway_station_scenarios/subway_station_max_ue_'
        New = 'subway_station_scenarios/subway_station_max_ue_'
    },
    @{
        Old = 'subway50'
        New = 'subway50'
    }
)

$totalChanged = 0
foreach ($file in $shFiles) {
    $original = Get-Content -Raw -LiteralPath $file.FullName
    $updated = $original
    foreach ($r in $replacements) {
        $updated = $updated.Replace($r.Old, $r.New)
    }

    if ($updated -ne $original) {
        $totalChanged++
        if ($DryRun) {
            Write-Host "[DRY] would update: $($file.Name)" -ForegroundColor Yellow
        }
        else {
            # Write back as UTF-8 without BOM, preserving LF if present.
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($updated)
            [System.IO.File]::WriteAllBytes($file.FullName, $bytes)
            Write-Host "updated: $($file.Name)" -ForegroundColor Green
        }
    }
    else {
        Write-Host "unchanged: $($file.Name)" -ForegroundColor DarkGray
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. $totalChanged file(s) would be modified." -ForegroundColor Cyan
}
else {
    Write-Host "Done. $totalChanged file(s) modified." -ForegroundColor Cyan
}
