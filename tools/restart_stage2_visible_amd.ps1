param(
    [string] $ManifestCsv = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\breed_runs\temporary_breed_run_full19_20260605_043214\manifest.trainable.csv",
    [string] $TaxonomyYaml = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\breed_runs\temporary_breed_run_full19_20260605_043214\taxonomy.trainable.yaml",
    [int] $Epochs = 10,
    [string] $StartSpecies = "",
    [string] $LogPath = "C:\abd_proj\results\reports\species_aware\train_two_stage_breed_models_efficientnetb2_temporary_breed_run_full19_20260605_043214_visible10.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv

Write-Host "Stage-2 AMD restart"
Write-Host "  repo:      $($state.RepoRoot)"
Write-Host "  python:    $($state.PythonExe)"
Write-Host "  manifest:  $ManifestCsv"
Write-Host "  taxonomy:  $TaxonomyYaml"
Write-Host "  epochs:    $Epochs"
if ($StartSpecies) {
    Write-Host "  start:     $StartSpecies"
}
Write-Host "  log:       $LogPath"
Write-Host ""

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null

Push-Location $state.RepoRoot
try {
    $env:HYDRA_FULL_ERROR = "1"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $args = @(
        "tools\train_two_stage_breed_models.py",
        "data=animal_breed_v0",
        "model/classifier=efficientnetb2",
        "data.split_manifest_path=$ManifestCsv",
        "data.taxonomy_path=$TaxonomyYaml",
        "model.classifier.train.epochs=$Epochs",
        "train.device=0"
    )
    if ($StartSpecies) {
        $args += "+start_species=$StartSpecies"
    }
    & $state.PythonExe @args 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ErrorActionPreference = $previousErrorActionPreference
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-2 training failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
    Write-Host "Stage-2 training finished successfully."
}
finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
}
