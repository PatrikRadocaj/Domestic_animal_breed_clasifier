param(
    [Parameter(Mandatory = $true)]
    [string] $RunStamp,
    [int] $PollSeconds = 60,
    [int] $TimeoutHours = 48
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv
$repoRoot = $state.RepoRoot
$reportRoot = Join-Path $repoRoot "results\reports\full_breed_pipeline\$RunStamp"
$breedRunName = "temporary_breed_run_retrain_$RunStamp"
$breedOutDir = Join-Path $repoRoot "data\processed\breed_runs\$breedRunName"
$watchLog = Join-Path $reportRoot "04b_label_taxonomy_audit_post_mining_watcher.log"
$auditLog = Join-Path $reportRoot "04b_label_taxonomy_audit_post_mining.log"
$auditOut = Join-Path $reportRoot "04b_label_taxonomy_audit_post_mining"
$pipelineLog = Join-Path $reportRoot "pipeline.log"

function Write-Watch {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $watchLog -Append
}

function Write-Pipeline {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $pipelineLog -Append
}

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Write-Watch "Watching for post-mining breed finalize outputs."
Write-Watch "run_stamp=$RunStamp"
Write-Watch "breed_out_dir=$breedOutDir"
Write-Watch "audit_out=$auditOut"

$required = @(
    (Join-Path $breedOutDir "manifest.trainable.csv"),
    (Join-Path $breedOutDir "taxonomy.trainable.yaml"),
    (Join-Path $breedOutDir "manifest.direct_yolo.csv"),
    (Join-Path $breedOutDir "taxonomy.direct_yolo.yaml"),
    (Join-Path $breedOutDir "label_review_report.csv")
)

$deadline = (Get-Date).AddHours($TimeoutHours)
while ((Get-Date) -lt $deadline) {
    $missing = @($required | Where-Object { -not (Test-Path $_) })
    if ($missing.Count -eq 0) {
        break
    }
    Write-Watch ("Waiting for finalize outputs. missing={0}" -f ($missing -join " | "))
    Start-Sleep -Seconds $PollSeconds
}

$missing = @($required | Where-Object { -not (Test-Path $_) })
if ($missing.Count -ne 0) {
    Write-Watch ("Timed out before post-mining audit. missing={0}" -f ($missing -join " | "))
    exit 2
}

Write-Watch "Finalize outputs detected. Waiting one poll interval so files settle."
Start-Sleep -Seconds $PollSeconds

New-Item -ItemType Directory -Force -Path $auditOut | Out-Null
Write-Pipeline "START label_taxonomy_audit_post_mining"
Write-Watch "Starting taxonomy audit."
Push-Location $repoRoot
try {
    & $state.PythonExe -u "tools\audit_label_taxonomy_qa.py" --out-dir $auditOut --processed-run-name $breedRunName 2>&1 | Tee-Object -FilePath $auditLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "label taxonomy audit failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
Write-Watch "Post-mining taxonomy audit completed."
Write-Pipeline "DONE label_taxonomy_audit_post_mining out=$auditOut"
