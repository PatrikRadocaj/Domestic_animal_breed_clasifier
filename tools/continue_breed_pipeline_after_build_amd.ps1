param(
    [string] $BreedRunName = "temporary_breed_run_full19_20260605_043214",
    [int] $ClassifierEpochs = 60,
    [int] $DetectorEpochs = 80,
    [int] $PollSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv
$repoRoot = "C:\abd_proj"
$pipelineLog = Join-Path $repoRoot "results\reports\species_aware\post_build_pipeline_$BreedRunName.log"
$breedOutDir = Join-Path $repoRoot "data\processed\breed_runs\$BreedRunName"
$manifestCsv = Join-Path $breedOutDir "manifest.csv"
$taxonomyYaml = Join-Path $breedOutDir "taxonomy.yaml"
$trainableManifestCsv = Join-Path $breedOutDir "manifest.trainable.csv"
$trainableTaxonomyYaml = Join-Path $breedOutDir "taxonomy.trainable.yaml"
$effectiveManifestCsv = $manifestCsv
$effectiveTaxonomyYaml = $taxonomyYaml
$preparedBreedDir = Join-Path $repoRoot "data\processed\yolo_breed_$BreedRunName"
$breedDataYaml = Join-Path $repoRoot "data\processed\yolo_breed_$BreedRunName.yaml"
$speciesBest = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\runs\detect\runs\detect\artifacts\hybrid_shared_detector\hybrid_shared_full19_bactrian_20more\weights\best.pt"

function Write-Log {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $pipelineLog -Value $line
}

function Invoke-PythonStep {
    param(
        [string] $Name,
        [string[]] $Args
    )
    Write-Log "START $Name"
    Push-Location $repoRoot
    try {
        & $state.PythonExe @Args 2>&1 | Tee-Object -FilePath $pipelineLog -Append
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    Write-Log "DONE $Name"
}

function Wait-ForBreedBuild {
    Write-Log "Waiting for breed build outputs in $breedOutDir"
    while ($true) {
        if ((Test-Path $manifestCsv) -and (Test-Path $taxonomyYaml)) {
            $manifestAge = ((Get-Date) - (Get-Item $manifestCsv).LastWriteTime).TotalSeconds
            if ($manifestAge -ge 60) {
                Write-Log "Breed build outputs detected."
                return
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $pipelineLog -Parent) | Out-Null
Write-Log "Post-build pipeline supervisor started."
Write-Log "AMD Python: $($state.PythonExe)"

Wait-ForBreedBuild

if ((Test-Path $trainableManifestCsv) -and (Test-Path $trainableTaxonomyYaml)) {
    $effectiveManifestCsv = $trainableManifestCsv
    $effectiveTaxonomyYaml = $trainableTaxonomyYaml
    $preparedBreedDir = Join-Path $repoRoot "data\processed\yolo_breed_${BreedRunName}_trainable"
    $breedDataYaml = Join-Path $repoRoot "data\processed\yolo_breed_${BreedRunName}_trainable.yaml"
    Write-Log "Using sanitized trainable breed manifest/taxonomy."
}
else {
    Write-Log "Using raw breed manifest/taxonomy."
}

Invoke-PythonStep -Name "prepare_yolo_breed_dataset" -Args @(
    "tools/prepare_yolo_dataset.py",
    "data=animal_breed_v0",
    "model/detector=yolo11_breed",
    "data.split_manifest_path=$effectiveManifestCsv",
    "data.taxonomy_path=$effectiveTaxonomyYaml",
    "data.prepared_yolo_breed_dir=$preparedBreedDir",
    "model.detector.data_yaml_path=$breedDataYaml",
    "model.detector.classes=breed"
)

Invoke-PythonStep -Name "train_two_stage_breed_models_efficientnetb2" -Args @(
    "tools/train_two_stage_breed_models.py",
    "data=animal_breed_v0",
    "model/classifier=efficientnetb2",
    "data.split_manifest_path=$effectiveManifestCsv",
    "data.taxonomy_path=$effectiveTaxonomyYaml",
    "model.classifier.train.epochs=$ClassifierEpochs",
    "train.device=0"
)

Invoke-PythonStep -Name "train_direct_breed_yolo" -Args @(
    "tools/train_detector.py",
    "data=animal_breed_v0",
    "model/detector=yolo11_breed",
    "data.split_manifest_path=$effectiveManifestCsv",
    "data.taxonomy_path=$effectiveTaxonomyYaml",
    "data.prepared_yolo_breed_dir=$preparedBreedDir",
    "model.detector.data_yaml_path=$breedDataYaml",
    "model.detector.base_model=$speciesBest",
    "model.detector.train.epochs=$DetectorEpochs",
    "train.device=0",
    "train.resume=false"
)

Write-Log "Post-build pipeline supervisor completed."
