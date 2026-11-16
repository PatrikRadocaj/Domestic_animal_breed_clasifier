param(
    [int] $Stage1Epochs = 50,
    [int] $Stage1Batch = 16,
    [int] $Stage1Workers = 16,
    [int] $Stage2Epochs = 10,
    [int] $DirectYoloEpochs = 50,
    [string] $RunStamp = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "retrain_full_breed_pipeline_visible_amd.ps1"
$repoRoot = Split-Path $PSScriptRoot -Parent
$wrapperDir = Join-Path $repoRoot "results\reports\full_breed_pipeline\launchers"
New-Item -ItemType Directory -Force -Path $wrapperDir | Out-Null
$wrapperStamp = if ([string]::IsNullOrWhiteSpace($RunStamp)) { Get-Date -Format "yyyyMMdd_HHmmss" } else { $RunStamp }
$wrapperPath = Join-Path $wrapperDir "launch_$wrapperStamp.ps1"

$wrapperLines = @(
    '$ErrorActionPreference = ''Stop'''
    "Set-Location '$repoRoot'"
    "& '$scriptPath' -Stage1Epochs $Stage1Epochs -Stage1Batch $Stage1Batch -Stage1Workers $Stage1Workers -Stage2Epochs $Stage2Epochs -DirectYoloEpochs $DirectYoloEpochs" +
        ($(if (-not [string]::IsNullOrWhiteSpace($RunStamp)) { " -RunStamp '$RunStamp'" } else { "" }))
    'Write-Host ""'
    'Write-Host "Pipeline window remains open."'
)
$wrapperLines | Set-Content -Path $wrapperPath -Encoding UTF8

$proc = Start-Process powershell.exe `
    -WorkingDirectory $repoRoot `
    -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $wrapperPath) `
    -PassThru

[PSCustomObject]@{
    Pid = $proc.Id
    Script = $scriptPath
    WorkingDirectory = $repoRoot
    RunStamp = $RunStamp
    Wrapper = $wrapperPath
} | ConvertTo-Json
