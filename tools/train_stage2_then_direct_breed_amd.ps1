param(
    [string] $ManifestCsv = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\breed_runs\temporary_breed_run_full19_20260605_043214\manifest.trainable.csv",
    [string] $TaxonomyYaml = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\breed_runs\temporary_breed_run_full19_20260605_043214\taxonomy.trainable.yaml",
    [string] $DataYaml = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\processed\yolo_breed_temporary_breed_run_full19_20260605_043214_trainable_rgb.yaml",
    [string] $BaseModel = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\runs\detect\runs\detect\artifacts\hybrid_shared_detector\hybrid_shared_full19_bactrian_20more\weights\best.pt",
    [int] $Stage2Epochs = 2,
    [int] $DirectEpochs = 20,
    [int] $DirectBatch = 4,
    [int] $DirectWorkers = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stage2Log = "C:\abd_proj\results\reports\species_aware\stage2_efficientnetb2_rerun_${timestamp}_e${Stage2Epochs}.log"
$directLog = "C:\abd_proj\results\reports\species_aware\direct_breed_yolo_rerun_${timestamp}_e${DirectEpochs}.log"
$runName = "yolo11_breed_species_seed_full19_trainable${DirectEpochs}_b${DirectBatch}_rgb_rerun_${timestamp}"

Write-Host "Sequence launch"
Write-Host "  manifest:    $ManifestCsv"
Write-Host "  taxonomy:    $TaxonomyYaml"
Write-Host "  data_yaml:   $DataYaml"
Write-Host "  base_model:  $BaseModel"
Write-Host "  stage2_log:  $stage2Log"
Write-Host "  direct_log:  $directLog"
Write-Host "  direct_run:  $runName"
Write-Host ""

& (Join-Path $PSScriptRoot "restart_stage2_visible_amd.ps1") `
    -ManifestCsv $ManifestCsv `
    -TaxonomyYaml $TaxonomyYaml `
    -Epochs $Stage2Epochs `
    -LogPath $stage2Log

& (Join-Path $PSScriptRoot "train_direct_breed_yolo_from_species_amd.ps1") `
    -ManifestCsv $ManifestCsv `
    -TaxonomyYaml $TaxonomyYaml `
    -DataYaml $DataYaml `
    -BaseModel $BaseModel `
    -Epochs $DirectEpochs `
    -Batch $DirectBatch `
    -Workers $DirectWorkers `
    -RunName $runName `
    -LogPath $directLog
