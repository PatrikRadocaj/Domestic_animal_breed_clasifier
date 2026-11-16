param(
    [string] $ManifestCsv = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\breed_runs\temporary_breed_run_full19_20260605_043214\manifest.trainable.csv",
    [string] $TaxonomyYaml = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\breed_runs\temporary_breed_run_full19_20260605_043214\taxonomy.trainable.yaml",
    [string] $DataYaml = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\yolo_breed_temporary_breed_run_full19_20260605_043214_trainable.yaml",
    [string] $BaseModel = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\runs\detect\runs\detect\artifacts\hybrid_shared_detector\hybrid_shared_full19_bactrian_20more\weights\best.pt",
    [int] $Epochs = 10,
    [int] $Batch = 2,
    [int] $Workers = 4,
    [string] $RunName = "yolo11_breed_species_seed_full19_trainable10_b2",
    [string] $LogPath = "C:\abd_proj\results\reports\species_aware\train_direct_breed_yolo_from_species_amd_full19_trainable10.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv

Write-Host "Direct-breed YOLO AMD launch"
Write-Host "  repo:      $($state.RepoRoot)"
Write-Host "  python:    $($state.PythonExe)"
Write-Host "  manifest:  $ManifestCsv"
Write-Host "  taxonomy:  $TaxonomyYaml"
Write-Host "  data_yaml: $DataYaml"
Write-Host "  base:      $BaseModel"
Write-Host "  epochs:    $Epochs"
Write-Host "  batch:     $Batch"
Write-Host "  workers:   $Workers"
Write-Host "  run:       $RunName"
Write-Host "  log:       $LogPath"
Write-Host ""

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null

Push-Location $state.RepoRoot
try {
    $env:HYDRA_FULL_ERROR = "1"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $args = @(
        "tools\train_detector.py",
        "data=animal_breed_v0",
        "model/detector=yolo11_breed",
        "data.split_manifest_path=$ManifestCsv",
        "data.taxonomy_path=$TaxonomyYaml",
        "model.detector.base_model=$BaseModel",
        "model.detector.data_yaml_path=$DataYaml",
        "model.detector.name=$RunName",
        "model.detector.train.epochs=$Epochs",
        "model.detector.train.batch=$Batch",
        "model.detector.train.workers=$Workers",
        "train.device=0"
    )
    & $state.PythonExe @args 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ErrorActionPreference = $previousErrorActionPreference
    if ($LASTEXITCODE -ne 0) {
        throw "Direct-breed YOLO training failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
    Write-Host "Direct-breed YOLO training finished successfully."
}
finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
}
