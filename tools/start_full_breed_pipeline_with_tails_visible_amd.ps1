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

$starter = Join-Path $PSScriptRoot "start_retrain_full_breed_pipeline_visible_amd.ps1"
$raw = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $starter `
    -Stage1Epochs $Stage1Epochs `
    -Stage1Batch $Stage1Batch `
    -Stage1Workers $Stage1Workers `
    -Stage2Epochs $Stage2Epochs `
    -DirectYoloEpochs $DirectYoloEpochs `
    -RunStamp $RunStamp

$payload = $raw | ConvertFrom-Json
$repoRoot = Split-Path $PSScriptRoot -Parent
$effectiveRunStamp = if ([string]::IsNullOrWhiteSpace($payload.RunStamp)) {
    Split-Path $payload.Wrapper -LeafBase -ErrorAction SilentlyContinue
} else {
    [string] $payload.RunStamp
}
if ($effectiveRunStamp -like "launch_*") {
    $effectiveRunStamp = $effectiveRunStamp.Substring(7)
}
$reportRoot = Join-Path $repoRoot "results\reports\full_breed_pipeline\$effectiveRunStamp"

$tailTargets = @(
    @{ Title = "QA/Pipeline"; Path = Join-Path $reportRoot "pipeline.log" },
    @{ Title = "Normalize"; Path = Join-Path $reportRoot "00_yolo_image_normalization.log" },
    @{ Title = "Stress"; Path = Join-Path $reportRoot "00_stage1_dataloader_stress.log" },
    @{ Title = "Stage1"; Path = Join-Path $reportRoot "01_stage1_species.log" },
    @{ Title = "BreedBuild"; Path = Join-Path $reportRoot "03_breed_build.log" },
    @{ Title = "Stage2"; Path = Join-Path $reportRoot "05_stage2_efficientnet.log" },
    @{ Title = "DirectYOLO"; Path = Join-Path $reportRoot "07_direct_breed_yolo.log" }
)

foreach ($target in $tailTargets) {
    $title = [string] $target.Title
    $path = [string] $target.Path
    $command = @(
        '$ErrorActionPreference = ''Stop'''
        "Write-Host 'Waiting for log: $path'"
        "while (!(Test-Path '$path')) { Start-Sleep -Seconds 2 }"
        "Write-Host 'Tailing: $path'"
        "Get-Content '$path' -Wait"
    ) -join "; "
    Start-Process powershell.exe `
        -WorkingDirectory $repoRoot `
        -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $command) `
        -WindowStyle Normal | Out-Null
}

[PSCustomObject]@{
    Pid = $payload.Pid
    RunStamp = $effectiveRunStamp
    ReportRoot = $reportRoot
    Wrapper = $payload.Wrapper
} | ConvertTo-Json
