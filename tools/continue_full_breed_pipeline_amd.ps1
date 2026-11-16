param(
    [string] $SpeciesRunDir = "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\runs\detect\runs\detect\artifacts\hybrid_shared_detector\hybrid_shared_full19_bactrian_20more",
    [string] $SpeciesLog = "C:\abd_proj\results\reports\species_aware\hybrid_full19_bactrian_20more_2026-06-05.log",
    [string] $BreedRunName = "temporary_breed_run_full19_20260605",
    [int] $ClassifierEpochs = 60,
    [int] $DetectorEpochs = 80,
    [int] $BreedBuildBatchSize = 32,
    [int] $PollSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv
$repoRoot = "C:\abd_proj"
$pipelineLog = Join-Path $repoRoot "results\reports\species_aware\full_breed_pipeline_$BreedRunName.log"
$resultsCsv = Join-Path $SpeciesRunDir "results.csv"
$bestPt = Join-Path $SpeciesRunDir "weights\best.pt"
$lastPt = Join-Path $SpeciesRunDir "weights\last.pt"
$breedOutDir = Join-Path $repoRoot "data\processed\breed_runs\$BreedRunName"
$manifestCsv = Join-Path $breedOutDir "manifest.csv"
$taxonomyYaml = Join-Path $breedOutDir "taxonomy.yaml"
$preparedBreedDir = Join-Path $repoRoot "data\processed\yolo_breed_$BreedRunName"
$breedDataYaml = Join-Path $repoRoot "data\processed\yolo_breed_$BreedRunName.yaml"

function Write-Log {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $pipelineLog -Append
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

function Wait-ForSpeciesRun {
    Write-Log "Waiting for species run to finish: $SpeciesRunDir"
    while ($true) {
        if ((Test-Path $resultsCsv) -and (Test-Path $bestPt) -and (Test-Path $lastPt)) {
            $rows = @(Import-Csv $resultsCsv)
            $ageSeconds = ((Get-Date) - (Get-Item $resultsCsv).LastWriteTime).TotalSeconds
            if ($rows.Count -ge 20 -and $ageSeconds -ge 90) {
                Write-Log "Species run appears complete with $($rows.Count) epochs recorded."
                return
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $pipelineLog -Parent) | Out-Null
Write-Log "Pipeline supervisor started."
Write-Log "Species log: $SpeciesLog"
Write-Log "AMD Python: $($state.PythonExe)"

Wait-ForSpeciesRun

Invoke-PythonStep -Name "build_temporary_breed_run_dataset" -Args @(
    "tools/build_temporary_breed_run_dataset.py",
    "--species-model", $bestPt,
    "--run-name", $BreedRunName,
    "--device", "0",
    "--batch-size", "$BreedBuildBatchSize"
)

Invoke-PythonStep -Name "prepare_yolo_breed_dataset" -Args @(
    "tools/prepare_yolo_dataset.py",
    "data=animal_breed_v0",
    "model/detector=yolo11_breed",
    "data.split_manifest_path=$manifestCsv",
    "data.taxonomy_path=$taxonomyYaml",
    "data.prepared_yolo_breed_dir=$preparedBreedDir",
    "model.detector.data_yaml_path=$breedDataYaml",
    "model.detector.classes=breed"
)

Invoke-PythonStep -Name "train_two_stage_breed_models_efficientnetb2" -Args @(
    "tools/train_two_stage_breed_models.py",
    "data=animal_breed_v0",
    "model/classifier=efficientnetb2",
    "data.split_manifest_path=$manifestCsv",
    "data.taxonomy_path=$taxonomyYaml",
    "model.classifier.train.epochs=$ClassifierEpochs",
    "train.device=auto"
)

Invoke-PythonStep -Name "train_direct_breed_yolo" -Args @(
    "tools/train_detector.py",
    "data=animal_breed_v0",
    "model/detector=yolo11_breed",
    "data.split_manifest_path=$manifestCsv",
    "data.taxonomy_path=$taxonomyYaml",
    "data.prepared_yolo_breed_dir=$preparedBreedDir",
    "model.detector.data_yaml_path=$breedDataYaml",
    "model.detector.base_model=$bestPt",
    "model.detector.train.epochs=$DetectorEpochs",
    "train.resume=false"
)

Write-Log "Pipeline supervisor completed."
